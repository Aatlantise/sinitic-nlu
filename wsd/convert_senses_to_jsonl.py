from pathlib import Path
import argparse
import csv
import json

def convert_csv_to_jsonl(input_path: Path, output_path: Path, source: str = "wsd"):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)

    with output_path.open("w", encoding="utf-8") as outf:
        for i, row in enumerate(rows):
            obj = {
                "id": i,
                "target": row.get("traditional", ""),
                "jyutping": row.get("jyutping", ""),
                "definition": row.get("definition", ""),
                "sentence1": row.get("sentence1", ""),
                "sentence1_source": row.get("sentence1_source", ""),
                "sentence2": row.get("sentence2", ""),
                "sentence2_source": row.get("sentence2_source", ""),
                "source": source,
            }
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a filtered dictionary senses CSV into JSONL with preserved example sentences"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="output/Filtered Dictionary Senses - cleaned.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "output_jsonl",
        nargs="?",
        default="output/Filtered Dictionary Senses - cleaned.jsonl",
        help="Path to the output JSONL file",
    )
    parser.add_argument(
        "-s",
        "--source",
        default="wsd",
        help="Dataset source label to include in each JSON object",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_csv)
    output_path = Path(args.output_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    convert_csv_to_jsonl(input_path, output_path, source=args.source)
    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
