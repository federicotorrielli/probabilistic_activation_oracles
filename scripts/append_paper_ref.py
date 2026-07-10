"""Append a paper reference + arxiv URL to EvilScript activation-oracle/taboo cards
whose body lacks https://arxiv.org/abs/2605.26045 — the paper-page index requires
a body URL, the YAML tag alone is not enough.

Usage:
  uv run --with huggingface_hub python scripts/append_paper_ref.py --dry
  uv run --with huggingface_hub python scripts/append_paper_ref.py
  uv run --with huggingface_hub python scripts/append_paper_ref.py --only EvilScript/<repo>
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, ModelCard

EXCLUDE = {"EvilScript/academic-sentiment-classifier"}
ARXIV_ID = "2605.26045"
ARXIV_URL = f"https://arxiv.org/abs/{ARXIV_ID}"
COMMIT_MSG = f"Add paper reference (arXiv:{ARXIV_ID}) to README body"

BLOCK = (
    "\n## Related Paper\n\n"
    "This adapter is one of the taboo target models used in "
    "[Confidence and Calibration of Activation Oracles for Reliable Interpretation of "
    f"Language Model Internals]({ARXIV_URL}) "
    f"(arXiv:{ARXIV_ID}).\n"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    api = HfApi()
    repos = list(api.list_models(author="EvilScript", full=True))
    candidates = [
        m.id
        for m in repos
        if ("activation-oracle" in m.id or "taboo" in m.id) and m.id not in EXCLUDE
    ]
    if args.only:
        candidates = [r for r in candidates if r == args.only]

    targets = []
    for rid in sorted(candidates):
        card = ModelCard.load(rid)
        if ARXIV_URL not in card.text:
            targets.append((rid, card))

    print(f"Candidates missing arxiv URL in body: {len(targets)}")

    updated = failed = 0
    for i, (rid, card) in enumerate(targets, 1):
        try:
            body = card.text or ""
            if not body.endswith("\n"):
                body += "\n"
            card.text = body + BLOCK
            if args.dry:
                print(f"  [{i}/{len(targets)}] {rid}: would append paper reference")
                updated += 1
                continue
            card.push_to_hub(rid, repo_type="model", commit_message=COMMIT_MSG)
            print(f"  [{i}/{len(targets)}] {rid}: appended paper reference")
            updated += 1
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(targets)}] {rid}: FAILED — {e}")

    print(f"\nupdated={updated} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
