#!/usr/bin/env python3
"""
Convert a hydrated DefExtra CSV into SciDef ground-truth JSON.

Input CSV columns (DefExtra schema):
  - paper_id, concept, definition, context, definition_type

Output JSON schema (SciDef ground truth):
  {
    "<paper_id>": {
      "<term>": {"definition": "...", "context": "...", "type": "explicit"}
    },
    ...
  }
"""

import argparse
import csv
import json
from pathlib import Path


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}


def _clean_text(val: str) -> str:
    return " ".join(str(val).strip().split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DefExtra hydrated CSV to SciDef ground-truth JSON.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to DefExtra hydrated CSV (with definition + context).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit to the first N unique papers (CSV order).",
    )
    parser.add_argument(
        "--max-defs-per-paper",
        type=int,
        default=None,
        help="Limit to the first N definitions per paper (CSV order).",
    )
    parser.add_argument(
        "--include-out-of-domain",
        action="store_true",
        help="Include rows marked as out_of_domain.",
    )
    parser.add_argument(
        "--lowercase-terms",
        action="store_true",
        help="Lowercase term keys in the output JSON.",
    )

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    data: dict[str, dict[str, dict[str, str]]] = {}
    skipped_missing = 0
    skipped_ood = 0
    skipped_dupe = 0
    total_rows = 0

    with args.csv.open("r", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            total_rows += 1
            if not args.include_out_of_domain and _truthy(
                row.get("is_out_of_domain", ""),
            ):
                skipped_ood += 1
                continue

            paper_id = _clean_text(row.get("paper_id", "")) or _clean_text(
                row.get("paper_id_left", ""),
            )
            term = _clean_text(
                row.get("concept", "") or row.get("term", ""),
            )
            definition = _clean_text(row.get("definition", ""))
            context = _clean_text(row.get("context", ""))
            def_type = _clean_text(
                row.get("definition_type", "") or row.get("type", ""),
            )

            if not paper_id or not term or not definition or not context:
                skipped_missing += 1
                continue

            if args.lowercase_terms:
                term = term.lower()

            if args.max_papers and paper_id not in data:
                if len(data) >= args.max_papers:
                    continue

            paper_defs = data.setdefault(paper_id, {})
            if args.max_defs_per_paper is not None:
                if len(paper_defs) >= args.max_defs_per_paper:
                    continue

            if term in paper_defs:
                skipped_dupe += 1
                continue

            paper_defs[term] = {
                "definition": definition,
                "context": context,
                "type": def_type or "explicit",
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print(
        "\n".join(
            [
                "DefExtra CSV → SciDef JSON conversion complete",
                f"  Input rows: {total_rows}",
                f"  Output papers: {len(data)}",
                f"  Skipped missing fields: {skipped_missing}",
                f"  Skipped out-of-domain: {skipped_ood}",
                f"  Skipped duplicate terms: {skipped_dupe}",
                f"  Output: {args.output}",
            ],
        ),
    )


if __name__ == "__main__":
    main()
