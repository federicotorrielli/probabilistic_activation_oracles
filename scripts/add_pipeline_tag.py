"""Add pipeline_tag: text-generation to EvilScript activation-oracle / taboo repos that lack it.

Usage:
  uv run --with huggingface_hub python scripts/add_pipeline_tag.py --dry
  uv run --with huggingface_hub python scripts/add_pipeline_tag.py
  uv run --with huggingface_hub python scripts/add_pipeline_tag.py --only EvilScript/<repo>
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, ModelCard

EXCLUDE = {"EvilScript/academic-sentiment-classifier"}
PIPELINE = "text-generation"
COMMIT_MSG = "Add pipeline_tag: text-generation"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    api = HfApi()
    repos = list(api.list_models(author="EvilScript", full=True))
    targets = [
        m
        for m in repos
        if ("activation-oracle" in m.id or "taboo" in m.id)
        and m.id not in EXCLUDE
        and m.pipeline_tag != PIPELINE
    ]
    if args.only:
        targets = [m for m in targets if m.id == args.only]
    print(f"Candidates: {len(targets)}")

    updated = unchanged = failed = 0
    for i, m in enumerate(targets, 1):
        rid = m.id
        try:
            card = ModelCard.load(rid)
            current = card.data.pipeline_tag
            if current == PIPELINE:
                unchanged += 1
                print(f"  [{i}/{len(targets)}] {rid}: already {PIPELINE}")
                continue
            card.data.pipeline_tag = PIPELINE
            if args.dry:
                print(
                    f"  [{i}/{len(targets)}] {rid}: would set pipeline_tag {current!r} -> {PIPELINE!r}"
                )
                updated += 1
                continue
            card.push_to_hub(rid, repo_type="model", commit_message=COMMIT_MSG)
            print(
                f"  [{i}/{len(targets)}] {rid}: set pipeline_tag {current!r} -> {PIPELINE!r}"
            )
            updated += 1
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(targets)}] {rid}: FAILED — {e}")

    print(f"\nupdated={updated} unchanged={unchanged} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
