"""Extract misclassified sample info into a standalone JSON file.

This script reads a training log JSON (created by `DebugLogger`) and writes a
new JSON containing only the misclassified sample information.

Example:
    python export_misclassified.py \
      --input logs/training_log_20260317_005331.json \
      --output logs/misclassified_20260317_005331.json

By default it will copy all fields in `misclassified_samples` unmodified.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Extract misclassified samples to a new JSON file")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the training log JSON file (output of DebugLogger)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the extracted misclassified JSON file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, only write the first N misclassified samples",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # The DebugLogger stores misclassified examples under `misclassified_samples`
    misclassified = data.get("misclassified_samples", [])

    if args.max_samples is not None:
        misclassified = misclassified[: args.max_samples]

    output_data = {
        "source_log": args.input,
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "n_misclassified": len(misclassified),
        "misclassified_samples": misclassified,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote {len(misclassified)} misclassified samples to: {args.output}")


if __name__ == "__main__":
    main()
