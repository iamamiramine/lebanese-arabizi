import json
import sys
from pathlib import Path
from enum import Enum


class OutputFormat(Enum):
    SHAREGPT = "sharegpt"
    JSONL_CONTENTS = "jsonl_contents"


def convert_to_sharegpt(entry):
    """
    Convert merged.json entry to ShareGPT format:
    {"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}
    """
    input_text = entry.get("input", "")
    output_text = entry.get("output", "")
    
    if not input_text or not output_text:
        return None
    
    return {
        "conversations": [
            {"from": "user", "value": input_text},
            {"from": "assistant", "value": output_text}
        ]
    }


def convert_to_jsonl_contents(entry):
    """
    Convert merged.json entry to JSONL contents format (like 2.jsonl):
    {"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
    """
    input_text = entry.get("input", "")
    output_text = entry.get("output", "")
    
    if not input_text or not output_text:
        return None
    
    return {
        "contents": [
            {"role": "user", "parts": [{"text": input_text}]},
            {"role": "model", "parts": [{"text": output_text}]}
        ]
    }


def convert_merged_file(input_path: Path, output_path: Path, output_format: OutputFormat):
    """
    Convert a merged.json file to the specified output format.
    """
    # Read input file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from {input_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read from {input_path}: {e}")
    
    # Handle both array and single object formats
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = [data]
    else:
        raise ValueError(f"Unexpected data format in {input_path}")
    
    # Convert entries
    converted_entries = []
    skipped = 0
    
    converter = None
    if output_format == OutputFormat.SHAREGPT:
        converter = convert_to_sharegpt
    elif output_format == OutputFormat.JSONL_CONTENTS:
        converter = convert_to_jsonl_contents
    
    for entry_num, entry in enumerate(entries, 1):
        try:
            converted = converter(entry)
            if converted:
                converted_entries.append(converted)
            else:
                skipped += 1
                print(
                    f"Warning: Skipped entry {entry_num} - missing input or output",
                    file=sys.stderr
                )
        except Exception as e:
            print(
                f"Warning: Failed to process entry {entry_num}: {e}",
                file=sys.stderr,
            )
            skipped += 1
            continue
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == OutputFormat.JSONL_CONTENTS:
        # Write as JSONL (one JSON object per line)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in converted_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    else:
        # Write as JSON array
        output_path.write_text(
            json.dumps(converted_entries, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    print(f"Converted {len(converted_entries)} entries from {input_path}")
    if skipped > 0:
        print(f"Skipped {skipped} entries")
    print(f"Output written to {output_path}")
    
    return converted_entries


def main(argv):
    if len(argv) < 3:
        print(
            "Usage: python scripts/convert_merged_to_formats.py <input.json> <format> [--out <output>]\n"
            "Formats: sharegpt | jsonl_contents",
            file=sys.stderr,
        )
        sys.exit(1)
    
    input_path = Path(argv[0])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Parse format
    format_str = argv[1].lower()
    try:
        output_format = OutputFormat(format_str)
    except ValueError:
        print(
            f"Unknown format: {format_str}. Valid formats: sharegpt, jsonl_contents",
            file=sys.stderr,
        )
        sys.exit(2)
    
    # Determine output path
    output_path = None
    i = 2
    while i < len(argv):
        if argv[i] == "--out":
            if i + 1 >= len(argv):
                print("--out requires a path", file=sys.stderr)
                sys.exit(2)
            output_path = Path(argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {argv[i]}", file=sys.stderr)
            sys.exit(2)
    
    # Generate default output path if not provided
    if output_path is None:
        if output_format == OutputFormat.SHAREGPT:
            output_path = input_path.with_stem(f"{input_path.stem}_sharegpt")
        elif output_format == OutputFormat.JSONL_CONTENTS:
            output_path = input_path.with_suffix(".jsonl")
    
    convert_merged_file(input_path, output_path, output_format)


if __name__ == "__main__":
    main(sys.argv[1:])

