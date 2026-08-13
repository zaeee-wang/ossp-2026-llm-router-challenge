# SPDX-FileCopyrightText: Copyright 2026 llm-budget-router contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the submitted router.

The operator re-runs the image with episode ids rotated and the batch order
permuted, then compares decisions per original id. Anything that leaks order,
position or identity into a routing decision shows up here rather than in the
audit, so these run in the standard `python -m unittest discover` sweep.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_submission,
)
from ossp_router.submission_router import load_bundled_artifact, main

REPO = Path(__file__).resolve().parents[1]
TOY = REPO / "data" / "toy" / "inputs.json"


def _run(input_path: Path, tier: str, out: Path) -> int:
    return main(["--input", str(input_path), "--tier", tier, "--output", str(out)])


def _decisions(path: Path) -> dict:
    sub = load_submission(path)
    return {d.episode_id: d.model_id for d in sub.decisions}


class TestBundledArtifact(unittest.TestCase):
    def test_artifact_is_bundled_and_matches_the_policy(self) -> None:
        artifact = load_bundled_artifact()
        policy = load_bundled_policy()
        self.assertEqual(artifact.policy_id, policy.policy_id)
        for tier in TIERS:
            self.assertIn(tier, artifact.tier_safety_ratios)

    def test_safety_ratios_leave_headroom(self) -> None:
        """A ratio at the limit passes on one split and fails on the next."""
        artifact = load_bundled_artifact()
        for tier in TIERS:
            self.assertLess(artifact.tier_safety_ratios[tier], 0.95, tier)


class TestRouterContract(unittest.TestCase):
    def test_produces_one_decision_per_episode(self) -> None:
        inputs = load_input(TOY)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.json"
            self.assertEqual(_run(TOY, "fast", out), 0)
            sub = load_submission(out)
            self.assertEqual(sub.tier, "fast")
            self.assertEqual(sub.split, inputs.split)
            self.assertEqual(sub.challenge_id, inputs.challenge_id)
            self.assertEqual(len(sub.decisions), len(inputs.episodes))
            self.assertEqual({d.episode_id for d in sub.decisions},
                             {e.episode_id for e in inputs.episodes})
            for d in sub.decisions:
                self.assertIn(d.model_id, MODEL_IDS)

    @unittest.skipIf(os.name != "posix", "POSIX permission bits only")
    def test_output_file_mode_is_0644(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.json"
            _run(TOY, "balanced", out)
            self.assertEqual(out.stat().st_mode & 0o777, 0o644)

    def test_no_leftover_files_in_the_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.json"
            _run(TOY, "premium", out)
            self.assertEqual([p.name for p in Path(tmp).iterdir()],
                             ["submission.json"])

    def test_repeated_runs_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            _run(TOY, "premium", a)
            _run(TOY, "premium", b)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_unknown_tier_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(["--input", str(TOY), "--tier", "nope",
                      "--output", str(Path(tmp) / "s.json")])


class TestOrderAndIdInvariance(unittest.TestCase):
    """The audit the operator performs, run locally on the public dev batch."""

    @classmethod
    def setUpClass(cls) -> None:
        materialized = REPO / "data" / "materialized" / "dev" / "inputs.json"
        cls.source = materialized if materialized.exists() else TOY

    def _permuted_copy(self, tmp: Path, rotate_ids: bool) -> tuple[Path, dict]:
        raw = json.loads(self.source.read_text(encoding="utf-8"))
        episodes = list(raw["episodes"])
        rng = random.Random(20260813)
        rng.shuffle(episodes)
        mapping = {}
        if rotate_ids:
            ids = [e["episode_id"] for e in episodes]
            rotated = ids[1:] + ids[:1]
            for episode, new_id in zip(episodes, rotated):
                mapping[new_id] = episode["episode_id"]
                episode["episode_id"] = new_id
        raw["episodes"] = episodes
        path = tmp / "permuted.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path, mapping

    def test_decisions_survive_shuffle_and_id_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_out = tmp / "base.json"
            _run(self.source, "premium", base_out)
            base = _decisions(base_out)

            shuffled, _ = self._permuted_copy(tmp, rotate_ids=False)
            shuffled_out = tmp / "shuffled.json"
            _run(shuffled, "premium", shuffled_out)
            self.assertEqual(_decisions(shuffled_out), base,
                             "decision changed when the batch was reordered")

            rotated, mapping = self._permuted_copy(tmp, rotate_ids=True)
            rotated_out = tmp / "rotated.json"
            _run(rotated, "premium", rotated_out)
            got = _decisions(rotated_out)
            remapped = {mapping[new_id]: model for new_id, model in got.items()}
            self.assertEqual(remapped, base,
                             "decision followed the episode id, not the prompt")


if __name__ == "__main__":
    unittest.main()
