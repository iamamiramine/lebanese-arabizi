import json
import sys
from pathlib import Path


def convert_jsonl_entry_to_merged_format(entry):
    """
    Convert a JSONL entry from format:
    {"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
    
    To merged.json format:
    {"instruction": "...", "input": "...", "output": "...", "history": []}
    """
    if "contents" not in entry:
        return None
    
    contents = entry["contents"]
    
    # Extract user input (first user message)
    user_text = None
    model_text = None
    
    for msg in contents:
        if msg.get("role") == "user":
            parts = msg.get("parts", [])
            if parts and isinstance(parts[0], dict):
                user_text = parts[0].get("text", "")
                if user_text:
                    break
    
    # Extract model output (first model message after user)
    for msg in contents:
        if msg.get("role") == "model":
            parts = msg.get("parts", [])
            if parts and isinstance(parts[0], dict):
                model_text = parts[0].get("text", "")
                if model_text:
                    break
    
    # Skip if we don't have both input and output
    if not user_text or not model_text:
        return None
    
    # Instruction field - leave empty as it requires manual annotation
    # The instruction describes what the conversation is about in English
    instruction = ""
    
    return {
        "instruction": instruction,
        "input": user_text,
        "output": model_text,
        "history": []
    }


def convert_jsonl_to_merged_format(input_path: Path, output_path: Path):
    """
    Convert a JSONL file to merged.json format.
    """
    converted_entries = []
    skipped = 0
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    converted = convert_jsonl_entry_to_merged_format(entry)
                    if converted:
                        converted_entries.append(converted)
                    else:
                        skipped += 1
                        print(
                            f"Warning: Skipped line {line_num} - missing user or model message",
                            file=sys.stderr
                        )
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse line {line_num} in {input_path}: {e}",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to read JSONL from {input_path}: {e}")
    
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
            "Usage: python scripts/convert_jsonl_to_merged.py <input.jsonl> [--out <output.json>]",
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
        # Default: same name but with .json extension
        output_path = input_path.with_suffix(".json")
    
    convert_jsonl_to_merged_format(input_path, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])

