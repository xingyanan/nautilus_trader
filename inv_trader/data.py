#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# <copyright file="data.py" company="Invariance Pte">
#  Copyright (C) 2018 Invariance Pte. All rights reserved.
#  The use of this source code is governed by the license as found in the LICENSE.md file.
#  http://www.invariance.com
# </copyright>
# -------------------------------------------------------------------------------------------------

import redis
import dateutil.parser

from decimal import Decimal
from redis import ConnectionError, StrictRedis
from typing import List

from inv_trader.enums import Resolution, QuoteType, Venue
from inv_trader.objects import Tick, Bar

# Private IP 10.135.55.111


class LiveDataClient:
    """
    Provides a live data client for alpha models and trading strategies.
    """

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 6379):
        """
        Initializes a new instance of the LiveDataClient class.

        :param host: The redis host IP address (default=127.0.0.1).
        :param port: The redis host port (default=6379).
        """
        self._host = host
        self._port = port
        self._client = None
        self._pubsub = None
        self._subscriptions_ticks = []
        self._subscriptions_bars = []

        self._regex_tick_delimiters = '. '

    # Temporary properties for development
    def client(self) -> StrictRedis:
        return self._client

    @property
    def is_connected(self) -> bool:
        """
        :return: True if the client is connected, otherwise false.
        """
        if self._client is None:
            return False

        try:
            self._client.ping()
        except ConnectionError:
            return False

        return True

    @property
    def subscriptions_all(self) -> dict:
        """
        :return: All subscribed channels as a dictionary direct from the Redis pub/sub.
        """
        return self._client.pubsub_channels()

    @property
    def subscriptions_ticks(self) -> List[str]:
        """
        :return: The list of tick channels subscribed to.
        """
        return self._subscriptions_ticks

    @property
    def subscriptions_bars(self) -> List[str]:
        """
        :return: The list of bar channels subscribed to.
        """
        return self._subscriptions_bars

    def connect(self) -> str:
        """
        Connect to the live database and create a local pub/sub server.
        """
        self._client = redis.StrictRedis(host=self._host, port=self._port, db=0)
        self._pubsub = self._client.pubsub()

        return f"Connected to live database at {self._host}:{self._port}."

    def disconnect(self) -> List[str]:
        """
        Disconnects from the local publish subscribe server and the database.
        """
        if self._client is None:
            return ["Disconnected (the client was never connected.)"]

        unsubscribed_tick = []
        unsubscribed_bars = []

        for tick_channel in self._subscriptions_ticks[:]:
            self._pubsub.unsubscribe(tick_channel)
            self._subscriptions_ticks.remove(tick_channel)
            unsubscribed_tick.append(tick_channel)

        for bar_channel in self._subscriptions_bars[:]:
            self._pubsub.unsubscribe(bar_channel)
            self._subscriptions_bars.remove(bar_channel)
            unsubscribed_bars.append(bar_channel)

        disconnect_message = [f"Unsubscribed from tick_data {unsubscribed_tick}.",
                              f"Unsubscribed from bars_data {unsubscribed_bars}."]

        self._client.connection_pool.disconnect()
        self._client = None
        self._pubsub = None
        self._subscriptions_ticks = []
        self._subscriptions_bars = []

        disconnect_message.append(f"Disconnected from live database at {self._host}:{self._port}.")
        return disconnect_message

    def subscribe_tick_data(
            self,
            symbol: str,
            venue: Venue) -> str:
        """
        Subscribe to live tick data for the given symbol and venue.

        :param symbol: The symbol for subscription.
        :param venue: The venue for subscription.
        """
        if symbol is None:
            raise ValueError("The symbol cannot be null.")
        if self._client is None:
            return "No connection has been established to the live database (please connect first)."
        if not self.is_connected:
            return "No connection is established with the live database."

        tick_channel = self._get_tick_channel(symbol, venue)

        self._pubsub.subscribe(**{tick_channel: self.hacked_tick_message_printer})
        #thread1 = self._pubsub.run_in_thread(sleep_time=0.001)

        if not any(tick_channel for s in self._subscriptions_ticks):
            self._subscriptions_ticks.append(tick_channel)
            self._subscriptions_ticks.sort()
            return f"Subscribed to {tick_channel}."
        return f"Already subscribed to {tick_channel}."

    def unsubscribe_tick_data(
            self,
            symbol: str,
            venue: Venue) -> str:
        """
        Unsubscribes from live tick data for the given symbol and venue.

        :param symbol: The symbol to unsubscribe from.
        :param venue: The venue to unsubscribe from.
        """
        if symbol is None:
            raise ValueError("The symbol cannot be null.")
        if self._client is None:
            return "No connection has been established to the live database (please connect first)."
        if not self.is_connected:
            return "No connection is established with the live database."

        tick_channel = self._get_tick_channel(symbol, venue)

        self._pubsub.unsubscribe(tick_channel)

        if any(tick_channel for s in self._subscriptions_ticks):
            self._subscriptions_ticks.remove(tick_channel)
            self._subscriptions_ticks.sort()
            return f"Unsubscribed from {tick_channel}."
        return f"Already unsubscribed from {tick_channel}."

    def subscribe_bar_data(
            self,
            symbol: str,
            venue: Venue,
            period: int,
            resolution: Resolution,
            quote_type: QuoteType) -> str:
        """
        Subscribe to live bar data for the given symbol and venue.

        :param symbol: The symbol for subscription.
        :param venue: The venue for subscription.
        :param period: The bar period for subscription (> 0).
        :param resolution: The bar resolution for subscription.
        :param quote_type: The bar quote type for subscription.
        """
        if symbol is None:
            raise ValueError("The symbol cannot be null.")
        if venue is None:
            raise ValueError("The venue cannot be null.")
        if period <= 0:
            raise ValueError("The period must be > 0.")
        if self._client is None:
            return "No connection has been established to the live database (please connect first)."
        if not self.is_connected:
            return "No connection is established with the live database."

        bar_channel = self._get_bar_channel(
            symbol,
            venue,
            period,
            resolution,
            quote_type)

        self._pubsub.subscribe(**{bar_channel: self.hacked_bar_message_printer})
        #thread2 = self._pubsub.run_in_thread(sleep_time=0.001)

        if not any(bar_channel for s in self._subscriptions_bars):
            self._subscriptions_bars.append(bar_channel)
            self._subscriptions_bars.sort()
            return f"Subscribed to {bar_channel}."
        return f"Already subscribed to {bar_channel}."

    def unsubscribe_bar_data(
            self,
            symbol: str,
            venue: Venue,
            period: int,
            resolution: Resolution,
            quote_type: QuoteType) -> str:
        """
        Unsubscribes from live bar data for the given symbol and venue.

        :param symbol: The symbol to unsubscribe from.
        :param venue: The venue to unsubscribe from.
        :param period: The bar period to unsubscribe from (> 0).
        :param resolution: The bar resolution to unsubscribe from.
        :param quote_type: The bar quote type to unsubscribe from.
        """
        if symbol is None:
            raise ValueError("The symbol cannot be null.")
        if venue is None:
            raise ValueError("The venue cannot be null.")
        if period <= 0:
            raise ValueError("The period must be > 0.")
        if self._client is None:
            return "No connection has been established to the live database (please connect first)."
        if not self.is_connected:
            return "No connection is established with the live database."

        bar_channel = self._get_bar_channel(
            symbol,
            venue,
            period,
            resolution,
            quote_type)

        self._pubsub.unsubscribe(bar_channel)

        if any(bar_channel for s in self._subscriptions_bars):
            self._subscriptions_bars.remove(bar_channel)
            self._subscriptions_bars.sort()
            return f"Unsubscribed from {bar_channel}."
        return f"Already unsubscribed from {bar_channel}."

    @staticmethod
    def _parse_tick(
            tick_channel: str,
            tick_string: str) -> Tick:
        """
        Parse a Tick object from the given UTF-8 string.

        :param tick_string: The channel the tick was received on.
        :param tick_string: The tick string.
        :return: The parsed tick object.
        """
        split_channel = tick_channel.split('.')
        split_tick = tick_string.split(',')

        return Tick(split_channel[0],
                    Venue[str(split_channel[1].upper())],
                    Decimal(split_tick[0]),
                    Decimal(split_tick[1]),
                    dateutil.parser.parse(split_tick[2]))

    @staticmethod
    def _parse_bar(
            bar_channel: str,
            bar_string: str) -> Bar:
        """
        Parse a Bar object from the given UTF-8 string.

        :param bar_channel: The channel the bar was received on.
        :param bar_string: The bar string.
        :return: The parsed bar object.
        """
        split_channel = bar_channel.split('.')
        split_channel2 = split_channel[1].split('-')
        split_bar = bar_string.split(',')

        print(split_channel)
        print(split_bar)

        return Bar(split_channel[0],
                   Venue[str(split_channel2[0].upper())],
                   Decimal(split_bar[0]),
                   Decimal(split_bar[1]),
                   Decimal(split_bar[2]),
                   Decimal(split_bar[3]),
                   long(split_bar[4]),
                   dateutil.parser.parse(split_bar[2]))

    @staticmethod
    def _get_tick_channel(
            symbol: str,
            venue: Venue) -> str:
        """
        Returns the tick channel name from the given parameters.
        """
        return f'{symbol}.{venue.name.lower()}'

    @staticmethod
    def _get_bar_channel(
            symbol: str,
            venue: Venue,
            period: int,
            resolution: Resolution,
            quote_type: QuoteType) -> str:
        """
        Returns the bar channel name from the given parameters.
        """
        return f'{symbol}.{venue.name.lower()}-{period}-{resolution.name.lower()}[{quote_type.name.lower()}]'

    def hacked_tick_message_printer(self, message):
        print(f"{message['channel']}: {message['data']}", end='\r')

    def hacked_bar_message_printer(self, message):
        print(f"{message['channel']}: {message['data']}", end='\r')
