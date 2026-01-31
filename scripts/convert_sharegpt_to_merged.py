import json
import sys
from pathlib import Path


def convert_sharegpt_entry_to_merged_format(entry):
    """
    Convert a ShareGPT entry from format:
    {"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}
    
    To merged.json format:
    {"instruction": "...", "input": "...", "output": "...", "history": []}
    """
    if "conversations" not in entry:
        return None
    
    conversations = entry["conversations"]
    
    # Extract user input (first user message)
    user_text = None
    assistant_text = None
    
    for msg in conversations:
        if msg.get("from") == "user":
            user_text = msg.get("value", "")
            if user_text:
                break
    
    # Extract assistant output (first assistant message after user)
    for msg in conversations:
        if msg.get("from") == "assistant":
            assistant_text = msg.get("value", "")
            if assistant_text:
                break
    
    # Skip if we don't have both input and output
    if not user_text or not assistant_text:
        return None
    
    # Instruction field - leave empty as it requires manual annotation
    # The instruction describes what the conversation is about in English
    instruction = ""
    
    return {
        "instruction": instruction,
        "input": user_text,
        "output": assistant_text,
        "history": []
    }


def convert_sharegpt_to_merged_format(input_path: Path, output_path: Path):
    """
    Convert a ShareGPT JSON file to merged.json format.
    """
    converted_entries = []
    skipped = 0
    
    try:
        # Read the JSON file
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both array and single object formats
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = [data]
        else:
            raise ValueError(f"Unexpected data format in {input_path}")
        
        for entry_num, entry in enumerate(entries, 1):
            try:
                converted = convert_sharegpt_entry_to_merged_format(entry)
                if converted:
                    converted_entries.append(converted)
                else:
                    skipped += 1
                    print(
                        f"Warning: Skipped entry {entry_num} - missing user or assistant message",
                        file=sys.stderr
                    )
            except Exception as e:
                print(
                    f"Warning: Failed to process entry {entry_num} in {input_path}: {e}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
                
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from {input_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read from {input_path}: {e}")
    
    # Write converted data to output file
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
            "Usage: python scripts/convert_sharegpt_to_merged.py <input.json> [--out <output.json>]",
            file=sys.stderr,
        )
        sys.exit(1)
    
    input_path = Path(argv[0])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if len(argv) >= 3 and argv[1] == "--out":
        output_path = Path(argv[2])
    else:
        # Default: same name but with _converted suffix
        output_path = input_path.with_stem(f"{input_path.stem}_converted")
    
    convert_sharegpt_to_merged_format(input_path, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])

