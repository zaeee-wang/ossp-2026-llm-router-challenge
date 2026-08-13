# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 llm-budget-router contributors
# SPDX-License-Identifier: Apache-2.0

"""Submission entry point: one tier, whole batch, bundled artifact.

Same CLI contract as the reference router (`--input --tier --output`), but the
model artifact is baked into the image rather than passed on the command line,
because the official run mounts only the input file.

Two departures from the public `hash_regex` baseline, both measured:

1. The log-cost heads carry an upper-quantile shift (z*sigma folded into the
   intercepts at training time). Costs are lognormal with very different
   spreads per model - the residual sd is 0.56 / 0.44 / 0.66 - so pricing the
   centre understates the heavy model relative to the light one. Shifting all
   three re-prices them consistently and lets the budget layer settle at a
   higher, still-safe operating point.

2. The tier safety ratios are chosen against the probability of exceeding the
   budget, not against the best score that happens to fit the public split.
   The published baseline sits at 99.6% of the premium limit, which passes on
   dev and fails about half the time on a resampled batch; these ratios keep
   every tier clear with a composition-shift margin on top.

Decisions depend only on prompt content: no episode id, split, batch position
or ordering is read anywhere in the path.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Optional, Sequence

from .hash_regex import HashRegexArtifact, make_hash_regex_submission, parse_artifact
from .heuristic import write_submission_atomic
from .protocol import (
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_policy,
)

ARTIFACT_RESOURCE = "router-artifact.v1.json"


def load_bundled_artifact() -> HashRegexArtifact:
    """Read the artifact shipped inside the image (no filesystem lookup)."""
    try:
        text = resources.read_text(
            "ossp_router.resources", ARTIFACT_RESOURCE, encoding="utf-8"
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise ProtocolError(f"번들 아티팩트를 찾을 수 없습니다: {exc}") from exc
    return parse_artifact(json.loads(text))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="예산 인식 프롬프트 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifact", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        artifact = (
            parse_artifact(json.loads(args.artifact.read_text(encoding="utf-8")))
            if args.artifact is not None
            else load_bundled_artifact()
        )
        plan = make_hash_regex_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: {args.tier} 제출 파일을 생성했습니다 "
        f"(예측 비용 비율 {plan.predicted_budget_ratio:.6f}, "
        f"안전계수 {plan.safety_ratio:.4f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
