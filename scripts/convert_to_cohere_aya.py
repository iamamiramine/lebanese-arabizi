import json
import sys
from pathlib import Path


def convert_entry_to_cohere_aya(entry):
    """
    Convert merged.json entry to Cohere Aya format:
    
    From:
    {"instruction": "", "input": "...", "output": "...", "history": []}
    
    To:
    {
      "inputs": "...",
      "targets": "..."
    }
    """
    input_text = entry.get("input", "")
    output_text = entry.get("output", "")
    
    if not input_text or not output_text:
        return None
    
    return {
        "inputs": input_text,
        "targets": output_text
    }


def convert_to_cohere_aya_format(input_path: Path, output_path: Path):
    """
    Convert a merged.json file to Cohere Aya format.
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
    
    for entry_num, entry in enumerate(entries, 1):
        try:
            converted = convert_entry_to_cohere_aya(entry)
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
    
    # Write output as JSON array
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    if len(argv) < 1:
        print(
            "Usage: python scripts/convert_to_cohere_aya.py <input.json> [--out <output.json>]",
            file=sys.stderr,
        )
        sys.exit(1)
    
    input_path = Path(argv[0])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    output_path = None
    i = 1
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
        output_path = input_path.with_stem(f"{input_path.stem}_cohere_aya")
    
    convert_to_cohere_aya_format(input_path, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])

