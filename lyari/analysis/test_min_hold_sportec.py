"""Tests for the MIN_HOLD re-derivation on Sportec. Standard library only:

    python -m unittest lyari.analysis.test_min_hold_sportec -v        (from the repo root)
"""

import os
import tempfile
import unittest

import numpy as np

from lyari.analysis import min_hold_sportec as M
from lyari.dataset import labels as L

H, A, D = L.H, L.A, L.D


def seq(*runs):
    return np.concatenate([np.full(n, t, np.int8) for t, n in runs])


EVENTS = """<?xml version="1.0"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:30:00.000+02:00">
 <KickOff GameSection="firstHalf"><Play Team="HOME"><Pass/></Play></KickOff>
</Event>
<Event EventTime="2023-05-27T15:30:02.000+02:00">
 <TacklingGame WinnerTeam="AWAY" WinnerRole="withoutBallControl" LoserTeam="HOME" LoserRole="withBallControl" WinnerResult="ballcontactSucceeded"/>
</Event>
<Event EventTime="2023-05-27T15:30:04.000+02:00">
 <OtherBallAction Team="AWAY" DefensiveClearance="true"/>
</Event>
<Event EventTime="2023-05-27T15:30:06.000+02:00">
 <BallClaiming Team="AWAY" Type="InterceptedBall"/>
</Event>
</PutDataRequest>
"""


class ReadEvents(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(EVENTS)

    def tearDown(self):
        os.remove(self.path)

    def test_kinds_and_teams(self):
        ev = M.read_events(self.path, "HOME")
        kinds = [(round(t - ev[0][0]), s, team, k) for t, s, team, k in ev]
        self.assertEqual(kinds, [(0, 1, -1, "kickoff"), (0, 1, 0, "pass"), (2, 1, 1, "touch"),
                                 (2, 1, 0, "control"), (4, 1, 1, "touch"), (6, 1, 1, "control")])

    def test_duel_loser_can_be_switched_off(self):
        ev = M.read_events(self.path, "HOME", duel_loser=False)
        self.assertNotIn((1, 0, "control"), [(s, team, k) for _, s, team, k in ev])


class Candidates(unittest.TestCase):
    def test_interruptions_only(self):
        flag = seq((D, 10), (A, 5), (H, 100), (A, 10), (H, 100), (A, 20), (D, 10), (H, 3), (A, 50))
        got = [(e - s, t) for s, e, t in M.candidates(flag)]
        # structure only, any length: A 5 and H 3 follow a dead ball (restarts) and are skipped
        self.assertEqual(got, [(100, H), (10, A), (100, H), (20, A), (50, A)])

    def test_matches_harmonise_at_infinite_hold(self):
        flag = seq((H, 100), (A, 10), (H, 60), (A, 30), (H, 100))
        cands = M.candidates(flag)
        out = L.harmonise(flag, 10 ** 6, max_rounds=1)
        for s, e, t in cands:
            self.assertTrue((out[s:e] != t).all())


class LabelCandidates(unittest.TestCase):
    def test_own_team_event_inside_window(self):
        cands = [(100, 120, A), (200, 220, A), (300, 320, H)]
        ev = {"pass": [(110, A), (210, H)], "control": [(323, H)]}
        np.testing.assert_array_equal(M.label_candidates(cands, ev, tol=5), [True, False, True])
        np.testing.assert_array_equal(M.label_candidates(cands, ev, tol=0), [True, False, False])


class Sync(unittest.TestCase):
    def test_recovers_shift(self):
        flag = seq((H, 250), (A, 250), (H, 250), (A, 250))
        period = np.ones(len(flag), np.int8)
        # passes stamped 0.4 s (10 frames) late relative to the flag
        events = [(f / 25.0 - 0.4, 1, int(flag[f]), "pass") for f in range(20, 1000, 37)]
        _, _, d, agree = M.sync(events, flag, period, 0.0, 1, shifts=np.arange(-30, 31))
        self.assertEqual(agree, 1.0)
        self.assertTrue(10 - 20 <= d <= 10 + 20)


class Threshold(unittest.TestCase):
    def test_balanced_accuracy_peaks_at_the_gap(self):
        dur = np.array([10, 20, 30, 80, 90, 100])
        ctrl = np.array([False, False, False, True, True, True])
        ba = [M.balanced_acc(dur, ctrl, h) for h in range(0, 120)]
        best = [h for h, v in enumerate(ba) if v == max(ba)]
        self.assertEqual((best[0], best[-1]), (31, 80))


class EventLabel(unittest.TestCase):
    def test_last_controlled_team_until_dead(self):
        flag = seq((H, 10), (A, 5), (H, 10), (D, 5), (A, 5))
        lab = M.event_label(flag, {"pass": [(2, H)], "control": [(20, A)]})
        np.testing.assert_array_equal(lab, seq((H, 20), (A, 5), (D, 5), (A, 5)))


if __name__ == "__main__":
    unittest.main()
