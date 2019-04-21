#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# <copyright file="test_position.py" company="Invariance Pte">
#  Copyright (C) 2018-2019 Invariance Pte. All rights reserved.
#  The use of this source code is governed by the license as found in the LICENSE.md file.
#  http://www.invariance.com
# </copyright>
# -------------------------------------------------------------------------------------------------

import unittest
import uuid

from decimal import Decimal

from inv_trader.common.clock import TestClock
from inv_trader.model.enums import Venue, OrderSide, MarketPosition
from inv_trader.model.objects import ValidString, Quantity, Symbol, Price, Money
from inv_trader.model.identifiers import GUID, OrderId, PositionId, ExecutionId, ExecutionTicket
from inv_trader.model.order import OrderFactory
from inv_trader.model.position import Position
from inv_trader.model.events import OrderPartiallyFilled, OrderFilled
from test_kit.stubs import TestStubs

UNIX_EPOCH = TestStubs.unix_epoch()
AUDUSD_FXCM = Symbol('AUDUSD', Venue.FXCM)
GBPUSD_FXCM = Symbol('GBPUSD', Venue.FXCM)


class PositionTests(unittest.TestCase):

    def setUp(self):
        # Fixture Setup
        self.order_factory = OrderFactory(
            id_tag_trader='001',
            id_tag_strategy='001',
            clock=TestClock())
        print('\n')

    def test_initialized_position_returns_expected_attributes(self):
        # Arrange
        # Act
        position = Position(
            AUDUSD_FXCM,
            PositionId('P123456'),
            UNIX_EPOCH)

        # Assert
        self.assertEqual(Quantity(0), position.quantity)
        self.assertEqual(MarketPosition.FLAT, position.market_position)
        self.assertEqual(0, position.event_count())
        self.assertEqual(None, position.last_execution_id)
        self.assertEqual(None, position.last_execution_ticket)
        self.assertTrue(position.is_flat)
        self.assertFalse(position.is_long)
        self.assertFalse(position.is_short)
        self.assertFalse(position.is_entered)
        self.assertFalse(position.is_exited)
        self.assertEqual(OrderSide.UNKNOWN, position.entry_direction)
        self.assertEqual(Decimal(0), position.points_realized)
        self.assertEqual(0.0, position.return_realized)
        self.assertEqual('Position(id=P123456) AUDUSD.FXCM FLAT', str(position))
        self.assertTrue(repr(position).startswith('<Position(id=P123456) AUDUSD.FXCM FLAT object at'))

    def test_position_filled_with_buy_order_returns_expected_attributes(self):
        # Arrange
        order = self.order_factory.market(
            AUDUSD_FXCM,
            OrderSide.BUY,
            Quantity(100000))

        position = Position(
            order.symbol,
            PositionId('P123456'),
            UNIX_EPOCH)

        order_filled = OrderFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            order.side,
            order.quantity,
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        # Act
        position.apply(order_filled)

        # Assert
        self.assertEqual(OrderId('O-19700101-000000-001-001-1'), position.from_order_id)
        self.assertEqual(Quantity(100000), position.quantity)
        self.assertEqual(MarketPosition.LONG, position.market_position)
        self.assertEqual(UNIX_EPOCH, position.entry_time)
        self.assertEqual(OrderSide.BUY, position.entry_direction)
        self.assertEqual(Price('1.00001'), position.average_entry_price)
        self.assertEqual(1, position.event_count())
        self.assertEqual([order.id], position.get_order_ids())
        self.assertEqual([ExecutionId('E123456')], position.get_execution_ids())
        self.assertEqual([ExecutionTicket('T123456')], position.get_execution_tickets())
        self.assertEqual(ExecutionId('E123456'), position.last_execution_id)
        self.assertEqual(ExecutionTicket('T123456'), position.last_execution_ticket)
        self.assertFalse(position.is_flat)
        self.assertTrue(position.is_long)
        self.assertFalse(position.is_short)
        self.assertTrue(position.is_entered)
        self.assertFalse(position.is_exited)
        self.assertEqual(Decimal(0), position.points_realized)
        self.assertEqual(0.0, position.return_realized)
        self.assertEqual(0.0004897053586319089, position.return_unrealized(Price('1.00050')))

    def test_position_filled_with_sell_order_returns_expected_attributes(self):
        # Arrange
        order = self.order_factory.market(
            AUDUSD_FXCM,
            OrderSide.SELL,
            Quantity(100000))

        position = Position(
            order.symbol,
            PositionId('P123456'),
            UNIX_EPOCH)

        order_filled = OrderFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            order.side,
            order.quantity,
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        # Act
        position.apply(order_filled)

        # Assert
        self.assertEqual(Quantity(100000), position.quantity)
        self.assertEqual(MarketPosition.SHORT, position.market_position)
        self.assertEqual(UNIX_EPOCH, position.entry_time)
        self.assertEqual(OrderSide.SELL, position.entry_direction)
        self.assertEqual(Price('1.00001'), position.average_entry_price)
        self.assertEqual(1, position.event_count())
        self.assertEqual(ExecutionId('E123456'), position.last_execution_id)
        self.assertEqual(ExecutionTicket('T123456'), position.last_execution_ticket)
        self.assertFalse(position.is_flat)
        self.assertFalse(position.is_long)
        self.assertTrue(position.is_short)
        self.assertTrue(position.is_entered)
        self.assertFalse(position.is_exited)
        self.assertEqual(Decimal(0), position.points_realized)
        self.assertEqual(0.0, position.return_realized)
        self.assertEqual(-0.0004897053586319089, position.return_unrealized(Price('1.00050')))

    def test_position_partial_fills_with_buy_order_returns_expected_attributes(self):
        # Arrange
        order = self.order_factory.market(
            AUDUSD_FXCM,
            OrderSide.BUY,
            Quantity(100000))

        position = Position(
            order.symbol,
            PositionId('P123456'),
            UNIX_EPOCH)

        order_partially_filled = OrderPartiallyFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            order.side,
            Quantity(50000),
            Quantity(50000),
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        # Act
        position.apply(order_partially_filled)
        position.apply(order_partially_filled)

        # Assert
        self.assertEqual(Quantity(100000), position.quantity)
        self.assertEqual(MarketPosition.LONG, position.market_position)
        self.assertEqual(UNIX_EPOCH, position.entry_time)
        self.assertEqual(OrderSide.BUY, position.entry_direction)
        self.assertEqual(Price('1.00001'), position.average_entry_price)
        self.assertEqual(2, position.event_count())
        self.assertEqual(ExecutionId('E123456'), position.last_execution_id)
        self.assertEqual(ExecutionTicket('T123456'), position.last_execution_ticket)
        self.assertFalse(position.is_flat)
        self.assertTrue(position.is_long)
        self.assertFalse(position.is_short)
        self.assertTrue(position.is_entered)
        self.assertFalse(position.is_exited)
        self.assertEqual(Decimal(0), position.points_realized)
        self.assertEqual(0.0, position.return_realized)
        self.assertEqual(Decimal('0.00049'), position.points_unrealized(Price('1.00050')))
        self.assertEqual(0.0004897053586319089, position.return_unrealized(Price('1.00050')))

    def test_position_partial_fills_with_sell_order_returns_expected_attributes(self):
        # Arrange
        order = self.order_factory.market(
            AUDUSD_FXCM,
            OrderSide.SELL,
            Quantity(100000))

        position = Position(
            order.symbol,
            PositionId('P123456'),
            UNIX_EPOCH)

        order_partially_filled = OrderPartiallyFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            order.side,
            Quantity(50000),
            Quantity(50000),
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        # Act
        position.apply(order_partially_filled)
        position.apply(order_partially_filled)

        # Assert
        self.assertEqual(Quantity(100000), position.quantity)
        self.assertEqual(MarketPosition.SHORT, position.market_position)
        self.assertEqual(UNIX_EPOCH, position.entry_time)
        self.assertEqual(OrderSide.SELL, position.entry_direction)
        self.assertEqual(Price('1.00001'), position.average_entry_price)
        self.assertEqual(2, position.event_count())
        self.assertEqual(ExecutionId('E123456'), position.last_execution_id)
        self.assertEqual(ExecutionTicket('T123456'), position.last_execution_ticket)
        self.assertFalse(position.is_flat)
        self.assertFalse(position.is_long)
        self.assertTrue(position.is_short)
        self.assertTrue(position.is_entered)
        self.assertFalse(position.is_exited)
        self.assertEqual(Decimal(0), position.points_realized)
        self.assertEqual(0.0, position.return_realized)
        self.assertEqual(Decimal('-0.00049'), position.points_unrealized(Price('1.00050')))
        self.assertEqual(-0.0004897053586319089, position.return_unrealized(Price('1.00050')))

    def test_position_filled_with_buy_order_then_sell_order_returns_expected_attributes(self):
        # Arrange
        order = self.order_factory.market(
            AUDUSD_FXCM,
            OrderSide.BUY,
            Quantity(100000))

        position = Position(
            order.symbol,
            PositionId('P123456'),
            UNIX_EPOCH)

        order_filled1 = OrderFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            OrderSide.BUY,
            order.quantity,
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        order_filled2 = OrderFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            OrderSide.SELL,
            order.quantity,
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        # Act
        position.apply(order_filled1)
        position.apply(order_filled2)

        # Assert
        self.assertEqual(Quantity(0), position.quantity)
        self.assertEqual(MarketPosition.FLAT, position.market_position)
        self.assertEqual(UNIX_EPOCH, position.entry_time)
        self.assertEqual(OrderSide.BUY, position.entry_direction)
        self.assertEqual(Price('1.00001'), position.average_entry_price)
        self.assertEqual(2, position.event_count())
        self.assertEqual(ExecutionId('E123456'), position.last_execution_id)
        self.assertEqual(ExecutionTicket('T123456'), position.last_execution_ticket)
        self.assertEqual(UNIX_EPOCH, position.exit_time)
        self.assertEqual(Price('1.00001'), position.average_exit_price)
        self.assertTrue(position.is_flat)
        self.assertFalse(position.is_long)
        self.assertFalse(position.is_short)
        self.assertTrue(position.is_entered)
        self.assertTrue(position.is_exited)
        self.assertEqual(Decimal(0), position.points_realized)  # No change in price
        self.assertEqual(Decimal(0), position.points_unrealized(Price('1.00050')))
        self.assertEqual(0.0, position.return_realized)  # No change in price
        self.assertEqual(0.0, position.return_unrealized(Price('1.00050')))

    def test_position_filled_with_sell_order_then_buy_order_returns_expected_attributes(self):
        # Arrange
        order = self.order_factory.market(
            AUDUSD_FXCM,
            OrderSide.SELL,
            Quantity(100000))

        position = Position(
            order.symbol,
            PositionId('P123456'),
            UNIX_EPOCH)

        order_filled1 = OrderFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            OrderSide.SELL,
            order.quantity,
            Price('1.00000'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        order_filled2 = OrderFilled(
            order.symbol,
            order.id,
            ExecutionId('E123456'),
            ExecutionTicket('T123456'),
            OrderSide.BUY,
            order.quantity,
            Price('1.00001'),
            UNIX_EPOCH,
            GUID(uuid.uuid4()),
            UNIX_EPOCH)

        # Act
        position.apply(order_filled1)
        position.apply(order_filled2)

        # Assert
        self.assertEqual(Quantity(0), position.quantity)
        self.assertEqual(MarketPosition.FLAT, position.market_position)
        self.assertEqual(UNIX_EPOCH, position.entry_time)
        self.assertEqual(OrderSide.SELL, position.entry_direction)
        self.assertEqual(Price('1.00000'), position.average_entry_price)
        self.assertEqual(2, position.event_count())
        self.assertEqual([order.id], position.get_order_ids())
        self.assertEqual(ExecutionId('E123456'), position.last_execution_id)
        self.assertEqual(ExecutionTicket('T123456'), position.last_execution_ticket)
        self.assertEqual(UNIX_EPOCH, position.exit_time)
        self.assertEqual(Price('1.00001'), position.average_exit_price)
        self.assertTrue(position.is_flat)
        self.assertFalse(position.is_long)
        self.assertFalse(position.is_short)
        self.assertTrue(position.is_entered)
        self.assertTrue(position.is_exited)
        self.assertEqual(Decimal('-0.00001'), position.points_realized)
        self.assertEqual(Decimal(0), position.points_unrealized(Price('1.00050')))  # No more quantity in market
        self.assertEqual(-1.001348027784843e-05, position.return_realized)
        self.assertEqual(0.0, position.return_unrealized(Price('1.00050')))  # No more quantity in market
