# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2024 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from scipy import stats
from numba import njit
#from numba.typed import Dict
import time
import datetime
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import List, Dict
from collections import deque
import os

from nautilus_trader.config import NonNegativeFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.core.rust.common import LogColor
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import book_type_from_str
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.events import PositionOpened
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.accounting.accounts.base import Account


class HighFrequencyGridTradingConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``HighFrequencyGridTrading`` instances.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    max_trade_size : Decimal
        The max position size per trade (volume on the level can be less).
    min_seconds_between_triggers : NonNegativeFloat, default 1.0
        The minimum time between triggers.
    book_type : str, default 'L2_MBP'
        The order book type for the strategy.
    use_quote_ticks : bool, default False
        If quote ticks should be used.
    subscribe_ticker : bool, default False
        If tickers should be subscribed to.
    use_trade_ticks : bool, default False
        If trade ticks should be used.
    grid_num : PositiveInt, default 20
        网格数.
    max_position : PositiveFloat, default xx
        算法的最大持仓.
    grid_interval : PositiveInt, default 10
        网格的间隔.
    half_spread : PositiveInt, default 20
        半价差.
    """

    instrument_id: InstrumentId
    max_trade_size: Decimal
    min_seconds_between_triggers: NonNegativeFloat = float(os.getenv('MIN_SECONDS_BETWEEN_TRIGGERS', 0.2))
    book_type: str = "L2_MBP"
    use_quote_ticks: bool = True
    subscribe_ticker: bool = True
    use_trade_ticks: bool = True

    grid_num: PositiveInt = int(os.getenv('GRID_NUM', 10))
    max_position: PositiveFloat = float(os.getenv('MAX_POSITION', 1.0))
    grid_interval: PositiveInt = int(os.getenv('GRID_INTERVAL', 28))
    half_spread: PositiveInt = int(os.getenv('HALF_SPREAD', 36))
    skew: PositiveFloat = float(os.getenv('SKEW', 1.2))

    looking_depth: PositiveFloat = float(os.getenv('LOOKING_DEPTH', 0.01))
    spread_min_threshold: PositiveInt = int(os.getenv('SPREAD_MIN_THRESHOLD', 2))
    spread_adjusted_factor: PositiveFloat = float(os.getenv('SPREAD_ADJUSTED_FACTOR', 1.2))
    interval_adjusted_factor: PositiveFloat = float(os.getenv('INTERVAL_ADJUSTED_FACTOR', 1.2))

    avoid_repeated_factor: PositiveFloat = float(os.getenv('AVOID_REPEATED_FACTOR', 3.0))
    
    # 0-False 1-True
    adjusted_spread_right_now: PositiveInt = int(os.getenv('ADJUSTED_SPREAD_RIGHT_NOW', 0))
    
    volatility_length: PositiveInt = int(os.getenv('VOLATILITY_LENGTH', 10))
    volatility_cal_freq: PositiveInt = int(os.getenv('VOLATILITY_CAL_FREQ', 25))
    mid_price_chg_length: PositiveInt = int(os.getenv('MID_PRICE_CHG_LENGTH', 300))


@njit
def linear_regression(x, y):
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept

@njit
def compute_coeff(xi, gamma, delta, A, k):
    inv_k = np.divide(1, k)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
    return c1, c2

@njit
def measure_trading_intensity(order_arrival_depth, out):
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue

        # Sets the tick index to 0 for the nearest possible best price
        # as the order arrival depth in ticks is measured from the mid-price
        tick = round(depth / .5) - 1

        # In a fast-moving market, buy trades can occur below the mid-price (and vice versa for sell trades)
        # since the mid-price is measured in a previous time-step;
        # however, to simplify the problem, we will exclude those cases.
        if tick < 0 or tick >= len(out):
            continue

        # All of our possible quotes within the order arrival depth,
        # excluding those at the same price, are considered executed.
        out[:tick] += 1

        max_tick = max(max_tick, tick)
    return out[:max_tick]


class HighFrequencyGridTrading(Strategy):
    """
    Cancels all orders and closes all positions on stop.

    Parameters
    ----------
    config : HighFrequencyGridTradingConfig
        The configuration for the instance.

    """

    def __init__(self, config: HighFrequencyGridTradingConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.max_trade_size = config.max_trade_size
        self.min_seconds_between_triggers = config.min_seconds_between_triggers
        self._last_trigger_timestamp: datetime.datetime | None = None
        self._last_check_timestamp: datetime.datetime | None = None
        self.instrument: Instrument | None = None
        self.book_type: BookType = book_type_from_str(self.config.book_type)
          
        self.init_balance: float = None
        self.order_side: OrderSide = None
        self.qty: float = None
        self.avg_px: float = None

        self.last_px: float = None
        self.last_order_side: OrderSide = None
        
        self.prev_mid: float = None

        self.volatility = deque(maxlen=config.volatility_length)
        self.mid_price_chg = deque(maxlen=config.mid_price_chg_length)

        self.t: int = 0

        out_dtype = np.dtype([
            ('half_spread_tick', 'f8'),
            ('skew', 'f8'),
            ('volatility', 'f8'),
            ('A', 'f8'),
            ('k', 'f8')
        ])

        self.arrival_depth = np.full(10_000_000, np.nan, np.float64)
        self.mid_price_chg = np.full(10_000_000, np.nan, np.float64)
        self.out = np.zeros(10_000_000, out_dtype)

        self.prev_mid_price_tick = np.nan
        self.mid_price_tick = np.nan

        self.tmp = np.zeros(1_000, np.float64)
        self.ticks = np.arange(len(self.tmp)) + 0.5

        self.A = np.nan
        self.k = np.nan
        self.volatility = np.nan
        self.gamma = 0.01
        self.delta = 1
        self.adj1 = 3.0
        self.adj2 = 1.0

        self.half_spread_tick = np.nan
        self.skew = np.nan

        self.trades = deque(maxlen=6_000)

    def on_start(self) -> None:
        """
        Actions to be performed on strategy start.
        """
        BINANCE = Venue("BINANCE")
        self.init_balance = float(self.portfolio.account(BINANCE).balance_free(Currency.from_str('USDT')))
        self.instrument = self.cache.instrument(self.instrument_id)
        self.tick_size: Price = self.instrument.price_increment

        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        if self.config.use_trade_ticks:
            self.subscribe_trade_ticks(instrument_id=self.instrument_id)

        if self.config.use_quote_ticks:
            self.subscribe_quote_ticks(self.instrument.id)
        
        self.subscribe_order_book_deltas(self.instrument.id, self.book_type)        
        
        self._last_trigger_timestamp = self.clock.utc_now()
        self._last_check_timestamp = self.clock.utc_now()
    
    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        """
        Actions to be performed when order book deltas are received.
        """
        pass

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Actions to be performed when a delta is received.
        """
        seconds_since_last_trigger = (
            self.clock.utc_now() - self._last_trigger_timestamp
        ).total_seconds()
        if seconds_since_last_trigger < self.min_seconds_between_triggers:
            self.log.debug("Time since last order < min_seconds_between_triggers - skipping")
            return 
        self.check_trigger()

    def on_trade_tick(self, tick: TradeTick) -> None:
        if np.isnan(self.mid_price_tick):
            self.log.info("No mid_price_tick")
            return

        self.trades.append(tick)

        depth = -np.inf
        trade_price_tick = tick.price / self.tick_size

        if tick.aggressor_side == OrderSide.BUY:
            depth = np.nanmax([trade_price_tick - Decimal(self.mid_price_tick), depth])
        else:
            depth = np.nanmax([Decimal(self.mid_price_tick) - trade_price_tick, depth])
        self.arrival_depth[self.t] = depth
        self.glft_mm()

    def glft_mm(self) -> None:
        if np.isnan(self.prev_mid_price_tick):
            self.prev_mid_price_tick = float(self.mid_price_tick)
            self.log.info("No prev_mid_price_tick")
            return

        # Records the mid-price change for volatility calculation.
        self.mid_price_chg[self.t] = Decimal(self.mid_price_tick) - Decimal(self.prev_mid_price_tick)

        if self.t % self.config.volatility_cal_freq == 0:
            # Window size is 10-minute.
            if self.t >= 3_000 - 1:
                # Calibrates A, k
                self.tmp[:] = 0
                lambda_ = measure_trading_intensity(self.arrival_depth[self.t + 1 - 3_000 : self.t + 1], self.tmp)
                
                if len(lambda_) > 2:
                    lambda_ = lambda_[:70] / 300
                    x = self.ticks[:len(lambda_)]
                    y = np.log(lambda_)
                    k_, logA = linear_regression(x, y)
                    self.A = np.exp(logA)
                    self.k = -k_

                # Updates the volatility.
                self.volatility = np.nanstd(self.mid_price_chg[self.t + 1 - 3_000 : self.t + 1]) * np.sqrt(5)
            
        self.t += 1
        self.prev_mid_price_tick = float(self.mid_price_tick)

        if self.t < 3_000:
            return

        # Computes bid price and ask price.
        c1, c2 = compute_coeff(self.gamma, self.gamma, self.delta, self.A, self.k)
        self.half_spread_tick = (c1 + self.delta / 2 * c2 * self.volatility)
        self.skew = c2 * self.volatility

        pct = stats.percentileofscore(self.arrival_depth[np.isfinite(self.arrival_depth)], self.half_spread_tick)
        your_pct = 100 - pct
        print('{:.4f}%'.format(your_pct))

        self.half_spread_tick *= self.adj1
        self.skew *= self.adj2

        # inverse of percentile
        pct = stats.percentileofscore(self.arrival_depth[np.isfinite(self.arrival_depth)], self.half_spread_tick)
        your_pct = 100 - pct
        print('adjust {:.4f}%'.format(your_pct))


    def on_order_book(self, order_book: OrderBook) -> None:
        """
        Actions to be performed when an order book update is received.
        """
        pass
    
    def check_trigger(self) -> None:
        """
        Check for trigger conditions.
        """
        self._last_trigger_timestamp = self.clock.utc_now()
        
        if not self.instrument:
            self.log.error("No instrument loaded")
            return

        # Fetch book from the cache being maintained by the `DataEngine`
        book = self.cache.order_book(self.instrument_id)
        if not book:
            self.log.error("No book being maintained")
            return

        if not book.spread():
            return

        best_bid_size: Quantity | None = book.best_bid_size()
        best_ask_size: Quantity | None = book.best_ask_size()
        if (best_bid_size is None or best_bid_size <= 0) or (best_ask_size is None or best_ask_size <= 0):
            self.log.warning("No market yet")
            return

        best_bid_price: Price | None = book.best_bid_price()
        best_ask_price: Price | None = book.best_ask_price()
        if (best_bid_price is None or best_bid_price <= 0) or (best_ask_price is None or best_ask_price <= 0):
            self.log.warning("No market yet")
            return
        
        if (best_ask_price - best_bid_price) / self.tick_size > self.config.spread_min_threshold:
            self.cancel_all_orders(self.instrument.id)
            return 

        net_position = self.portfolio.net_position(self.instrument_id)
        mid_price = (best_bid_price + best_ask_price) / Decimal('2.0')

        mid_price_tick = mid_price / self.tick_size
        self.mid_price_tick = float(mid_price_tick)

        if np.isnan(self.half_spread_tick) or np.isnan(self.skew):
            self.log.info(
                f"Waiting for half_spread and skew to warm up [{self.t}]",
                color=LogColor.BLUE,
            )
            return

        skew_position = np.power(self.config.skew, 1+float(net_position) / self.config.max_position)
        reservation_price = mid_price - self.tick_size * Decimal(skew_position)
        
        if self.prev_mid is None:
            self.prev_mid = mid_price
            return 

        if reservation_price == self.prev_mid:
            return 

        self.mid_price_chg.append(mid_price - self.prev_mid)

        looking_depth = int(np.floor(float(mid_price) * self.config.looking_depth / self.tick_size))
        grid_interval = max(self.config.grid_interval, int(np.floor(looking_depth/self.config.grid_num)))
        
        # Update volatility every 5 seconds. 
        if self.t % self.config.volatility_cal_freq == 0:
            # Window size is 1-minute.
            if self.t >= self.config.mid_price_chg_length - 1:
                # Updates the volatility.
                volatility = np.nanstd(self.mid_price_chg) * np.sqrt(1.0/self.min_seconds_between_triggers)
        
        self.volatility.append(volatility)
        
        volatility_chg_ratio = 1.0
        if len(self.volatility) > 1:
            volatility_chg_ratio = np.maximum(volatility / (self.volatility[-2]+1e-5), 1.0)
        
        grid_interval *= self.tick_size
        bid_half_spread = self.tick_size * self.config.half_spread * volatility_chg_ratio
        ask_half_spread = self.tick_size * self.config.half_spread * volatility_chg_ratio

        pow_coef = 0.0
        if self.portfolio.is_net_short(self.instrument_id):
            pow_coef = np.minimum(0, float(net_position) + self.config.max_position / 2)
        elif self.portfolio.is_net_long(self.instrument_id):
            pow_coef = np.maximum(0, float(net_position) - self.config.max_position / 2)
        
        if self.config.adjusted_spread_right_now == 1:
            bid_half_spread *= Decimal(np.power(self.config.spread_adjusted_factor, net_position / self.config.max_position))
            ask_half_spread *= Decimal(np.power(1/self.config.spread_adjusted_factor, net_position / self.config.max_position))
        else:
            bid_half_spread *= Decimal(np.power(self.config.spread_adjusted_factor, pow_coef / self.config.max_position))
            ask_half_spread *= Decimal(np.power(1/self.config.spread_adjusted_factor, pow_coef / self.config.max_position))

        # Since our price is skewed, it may cross the spread. To ensure market making and avoid crossing the spread,
        # limit the price to the best bid and best ask.
        bid_price = np.minimum(reservation_price - bid_half_spread, best_bid_price)
        ask_price = np.maximum(reservation_price + ask_half_spread, best_ask_price)

        # Aligns the prices to the grid.
        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        if len(self.cache.orders_inflight(strategy_id=self.id)) > 0:
            self.log.info("Already have orders in flight - skipping.")
            return
        
        fib_coef = [1.,1.,2.,3.,5.,8.,13.,21.,34.,55.]
        
        half_spread = self.tick_size * self.config.half_spread
        new_bid_orders = dict()
        if net_position < self.config.max_position:
            for coef, i in zip(fib_coef, range(self.config.grid_num)):
                bid_price_tick = round(bid_price / self.tick_size)
                
                if net_position != 0 and self.last_px is not None and self.last_order_side is not None:
                    if self.last_order_side == OrderSide.BUY and \
                            bid_price > self.last_px - Decimal(self.config.avoid_repeated_factor) * half_spread and \
                            bid_price < self.last_px + Decimal(self.config.avoid_repeated_factor) * half_spread:
                        bid_price -= Decimal(coef) * grid_interval
                        continue

                # order price in tick is used as order id.
                new_bid_orders[np.uint64(bid_price_tick)] = bid_price
                
                if net_position > self.config.max_position/2:
                    bid_price -= Decimal(coef) * grid_interval * Decimal(np.power(self.config.interval_adjusted_factor, pow_coef / self.config.max_position))
                else:
                    bid_price -= Decimal(coef) * grid_interval

        new_ask_orders = dict()
        if -net_position < self.config.max_position:
            for coef, i in zip(fib_coef, range(self.config.grid_num)):
                ask_price_tick = round(ask_price / self.tick_size)
                
                if net_position != 0 and self.last_px is not None and self.last_order_side is not None:
                    if self.last_order_side == OrderSide.SELL and \
                            ask_price > self.last_px - Decimal(self.config.avoid_repeated_factor) * half_spread and \
                            ask_price < self.last_px + Decimal(self.config.avoid_repeated_factor) * half_spread:
                        ask_price += Decimal(coef) * grid_interval
                        continue

                # order price in tick is used as order id.
                new_ask_orders[np.uint64(ask_price_tick)] = ask_price
                
                if net_position < -self.config.max_position/2:
                    ask_price += Decimal(coef) * grid_interval * Decimal(np.power(1/self.config.interval_adjusted_factor, pow_coef / self.config.max_position))
                else:
                    ask_price += Decimal(coef) * grid_interval
        
        open_orders = self.cache.orders_open(instrument_id=self.instrument_id)
        for order in open_orders:
            if order.status != OrderStatus.CANCELED and order.is_open:
                price_tick = np.uint64(round(order.price / self.tick_size))
                if order.side == OrderSide.BUY and price_tick not in new_bid_orders:
                    self.cancel_order(order)
                elif order.side == OrderSide.SELL and price_tick not in new_ask_orders:
                    self.cancel_order(order)
        
        orders = self.cache.orders_open(instrument_id=self.instrument_id)
        for order_id, order_price in new_bid_orders.items():
            # Posts a new buy order if there is no working order at the price on the new grid.
            open_order_price_ticks = []
            for open_order in orders:
                if open_order.side == OrderSide.BUY:
                    bid_price_tick = round(open_order.price / self.tick_size)
                    open_order_price_ticks.append(np.uint64(bid_price_tick))

            if order_id not in open_order_price_ticks:
                self.buy(order_price, self.max_trade_size)
                

        for order_id, order_price in new_ask_orders.items():
            # Posts a new sell order if there is no working order at the price on the new grid.
            open_order_price_ticks = []
            for open_order in orders:
                if open_order.side == OrderSide.SELL:
                    ask_price_tick = round(open_order.price / self.tick_size)
                    open_order_price_ticks.append(np.uint64(ask_price_tick))

            if order_id not in open_order_price_ticks:
                self.sell(order_price, self.max_trade_size)

        self.prev_mid = reservation_price
  
        # Position management, if the current position exceeds the maximum position, then start to reduce a small amount
        # of the position, so that the position is dynamically maintained below the maximum position. The backtesting
        # results show better optimization for both drawdown and returns.
        if net_position >= self.config.max_position:
            self.sell(best_ask_price, self.max_trade_size)
        elif net_position <= -self.config.max_position:
            self.buy(best_bid_price, self.max_trade_size)

        self.t += 1

    def buy(self, bid_price, quantity) -> None:
        order = self.order_factory.limit(
            instrument_id=self.instrument.id,
            price=self.instrument.make_price(bid_price),
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            #time_in_force=TimeInForce.GTD,
            #expire_time=self.clock.utc_now() + pd.Timedelta(seconds=10),
            #post_only=True,  # default value is True
            #reduce_only=False
        )
        #self.log.info(f"Hitting! {order=}", color=LogColor.BLUE)
        self.submit_order(order)
    
    def sell(self, ask_price, quantity)-> None:
        order = self.order_factory.limit(
            instrument_id=self.instrument.id,
            price=self.instrument.make_price(ask_price),
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
            #time_in_force=TimeInForce.GTD,
            #expire_time=self.clock.utc_now() + pd.Timedelta(seconds=10),
            #post_only=True,  # default value is True
            #reduce_only=False
        )
            
        #self.log.info(f"Hitting! {order=}", color=LogColor.BLUE)
        self.submit_order(order)

    def on_event(self, event: Event) -> None:
        if isinstance(event, PositionOpened):
            self.qty = event.signed_qty
            self.order_side = event.side
            self.avg_px = event.avg_px_open

        if isinstance(event, OrderFilled):
            self.last_px = event.last_px
            self.last_order_side = event.order_side
        
    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        if self.instrument is None:
            return

        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
