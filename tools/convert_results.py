#!/usr/bin/env python3
"""Convert a benchmark results JSON file from the old format (without percentage
scores) to the new format. Adds percentage fields alongside existing raw scores
and backfills missing metadata fields. Does not modify any raw score values."""

import argparse
import json
from pathlib import Path


def score_to_percentage(score):
    """Convert a 1–5 score to a 0–100 percentage. Returns None if score is None."""
    if score is None:
        return None
    return round((score - 1) / 4 * 100, 2)


def convert(data):
    """Add percentage fields to a benchmark results dict in place."""
    # --- Per-language results ---
    for lang, lang_data in data.get("results", {}).items():
        # Per-question percentages
        for question in lang_data.get("questions", []):
            if "mean_score_percentage" not in question:
                question["mean_score_percentage"] = score_to_percentage(question.get("mean_score"))

        # Per-language percentage
        if "average_score_percentage" not in lang_data:
            lang_data["average_score_percentage"] = score_to_percentage(lang_data.get("average_score"))

    # --- Summary ---
    summary = data.get("summary", {})
    if "overall_average_score_percentage" not in summary:
        summary["overall_average_score_percentage"] = score_to_percentage(summary.get("overall_average_score"))

    for lang_summary in summary.get("languages", {}).values():
        if "average_score_percentage" not in lang_summary:
            lang_summary["average_score_percentage"] = score_to_percentage(lang_summary.get("average_score"))

    # --- Metadata backfill ---
    metadata = data.get("metadata", {})
    if "limit" not in metadata:
        metadata["limit"] = None
    if "files_expected" not in metadata:
        metadata["files_expected"] = len(data.get("results", {}))
    if "files_completed" not in metadata:
        metadata["files_completed"] = len(data.get("results", {}))

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convert old benchmark results JSON to new format with percentage scores."
    )
    parser.add_argument("--input", required=True, dest="input_path", help="Input JSON file path")
    parser.add_argument("--output", required=True, dest="output_path", help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    convert(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Converted results written to {output_path}")


if __name__ == "__main__":
    main()
