"""
Regression test for the live data-feed stop→start lifecycle.

Root cause this guards against: KiteTicker runs on Twisted's reactor, which
cannot be restarted once stopped within the same process. The old DataFeed.stop()
called ws.stop() (reactor.stop()), so a later DataFeed.start() raised
twisted.internet.error.ReactorNotRestartable and left a dead ticker thread —
the bot reported bot_running=True but received zero ticks.

The fix keeps the reactor/connection alive across stop→start and treats stop as
"pause tick processing" rather than "tear down the reactor". These tests verify:
  * stop() never stops the reactor (ws.stop() is never called),
  * start→stop→start does not construct a second ticker / relaunch the reactor,
  * ticks are processed while running, dropped while paused, processed again
    after resume,
  * a redundant start() while already running is a safe no-op.

Run standalone (no pytest needed):
    .venv/bin/python -m unittest backend.test_data_feed -v
  or
    cd backend && ../.venv/bin/python -m unittest test_data_feed -v

──────────────────────────────────────────────────────────────────────────────
Manual repro of the original bug (for reference — do NOT need live creds for the
unit test above):
  1. Authenticate Kite, POST /bot/start.  Confirm /indicators nifty_price > 0.
  2. POST /bot/stop, then POST /bot/start again within a few seconds.
  3. Old behaviour: Cloud Run logs show "Connection closed: 1006" then
     "twisted.internet.error.ReactorNotRestartable" + "Exception in thread";
     /status keeps bot_running=True while nifty_price stays 0.0.
  4. Fixed behaviour: stop logs "DataFeed paused (connection kept alive)", the
     second start logs "DataFeed resumed (reactor kept alive)", no traceback,
     ticks resume and nifty_price advances again.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Make backend/ importable whether run from repo root or from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_feed  # noqa: E402
from config import NIFTY_INDEX_TOKEN  # noqa: E402


class FakeTicker:
    """Stand-in for KiteTicker that records lifecycle calls and never touches
    a real reactor. is_connected() returns True so the resume path treats the
    connection as alive (no reconnect needed)."""

    instances = []

    def __init__(self, api_key, access_token):
        self.api_key = api_key
        self.access_token = access_token
        self.connect_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.on_ticks = None
        self.on_connect = None
        self.on_close = None
        self.on_error = None
        FakeTicker.instances.append(self)

    # connect() is invoked as the thread target in DataFeed.start().
    def connect(self, threaded=False, **kwargs):
        self.connect_calls += 1

    def is_connected(self):
        return True

    def stop(self):
        # This is the dangerous call (reactor.stop()) that must never happen.
        self.stop_calls += 1

    def close(self, *a, **k):
        self.close_calls += 1

    # Helper for tests to push a NIFTY tick through the registered handler.
    def push_nifty_tick(self, price):
        self.on_ticks(self, [{"instrument_token": NIFTY_INDEX_TOKEN,
                              "last_price": price}])


class DataFeedLifecycleTest(unittest.TestCase):

    def setUp(self):
        FakeTicker.instances.clear()
        self.feed = data_feed.DataFeed()
        fake_kite = SimpleNamespace(api_key="k", access_token="t")
        # auth and kiteconnect are imported lazily inside DataFeed.start();
        # patch them at their source modules so the import picks up the fakes.
        self._patches = [
            mock.patch("auth.get_kite", return_value=fake_kite, create=True),
            mock.patch("kiteconnect.KiteTicker", FakeTicker, create=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_stop_start_keeps_single_reactor_and_never_stops_it(self):
        # First start launches exactly one ticker.
        self.feed.start()
        self.assertTrue(self.feed.is_running)
        self.assertEqual(len(FakeTicker.instances), 1)
        ws = FakeTicker.instances[0]

        # Ticks flow into the builder while running.
        ws.push_nifty_tick(100.0)
        self.assertEqual(self.feed.last_price, 100.0)

        # Stop = pause. Reactor must NOT be stopped.
        self.feed.stop()
        self.assertFalse(self.feed.is_running)
        self.assertEqual(ws.stop_calls, 0)

        # Ticks are dropped while paused (price stays frozen).
        ws.push_nifty_tick(200.0)
        self.assertEqual(self.feed.last_price, 100.0)

        # Restart = resume on the SAME ticker; no second instance, no reactor relaunch.
        self.feed.start()
        self.assertTrue(self.feed.is_running)
        self.assertEqual(len(FakeTicker.instances), 1)
        self.assertEqual(ws.connect_calls, 1)   # reactor thread launched once

        # Ticks flow again after resume.
        ws.push_nifty_tick(300.0)
        self.assertEqual(self.feed.last_price, 300.0)

        # The reactor was never stopped across the whole cycle.
        self.assertEqual(ws.stop_calls, 0)

    def test_redundant_start_is_noop(self):
        self.feed.start()
        self.feed.start()   # already running → no-op
        self.assertEqual(len(FakeTicker.instances), 1)
        self.assertEqual(FakeTicker.instances[0].connect_calls, 1)

    def test_stop_before_start_is_safe(self):
        # Never started: stop should be a harmless no-op.
        self.feed.stop()
        self.assertFalse(self.feed.is_running)
        self.assertEqual(len(FakeTicker.instances), 0)

    def test_repeated_stop_start_cycles(self):
        self.feed.start()
        ws = FakeTicker.instances[0]
        for _ in range(5):
            self.feed.stop()
            self.feed.start()
        self.assertEqual(len(FakeTicker.instances), 1)
        self.assertEqual(ws.stop_calls, 0)
        self.assertEqual(ws.connect_calls, 1)
        self.assertTrue(self.feed.is_running)


if __name__ == "__main__":
    unittest.main(verbosity=2)
