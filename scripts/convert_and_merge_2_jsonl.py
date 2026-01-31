import json
import sys
from pathlib import Path


def load_jsonl(path: Path):
    """Load JSONL file (one JSON object per line)."""
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse line {line_num} in {path}: {e}",
                        file=sys.stderr,
                    )
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to read JSONL from {path}: {e}")
    return data


def convert_jsonl_contents_to_merged(entry):
    """Convert JSONL contents format to merged format."""
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
    
    return {
        "instruction": "",
        "input": user_text,
        "output": model_text,
        "history": []
    }


def convert_merged_to_sharegpt(entry):
    """Convert merged format to ShareGPT format."""
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


def convert_merged_to_sharegptgemma(entry):
    """Convert merged format to ShareGPTGemma format."""
    input_text = entry.get("input", "")
    output_text = entry.get("output", "")
    
    if not input_text or not output_text:
        return None
    
    return {
        "conversations": [
            {"content": input_text, "role": "user"},
            {"content": output_text, "role": "assistant"}
        ]
    }


def convert_merged_to_jsonl_contents(entry):
    """Convert merged format to JSONL contents format."""
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


def convert_merged_to_cohere_aya(entry):
    """Convert merged format to Cohere Aya format."""
    input_text = entry.get("input", "")
    output_text = entry.get("output", "")
    
    if not input_text or not output_text:
        return None
    
    return {
        "inputs": input_text,
        "targets": output_text
    }


def load_json_array(path: Path):
    """Load JSON array or object from file."""
    if not path.exists():
        return []
    
    try:
        text = path.read_text(encoding="utf-8")
        text_stripped = text.strip()
        data = json.loads(text_stripped)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return []


def save_json(path: Path, data):
    """Save data as JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def save_jsonl(path: Path, data):
    """Save data as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    """Main function to convert 2.jsonl to all formats and merge."""
    base_dir = Path("data/fine_tuning")
    
    # Input file
    input_file = base_dir / "2.jsonl"
    if not input_file.exists():
        print(f"Error: {input_file} not found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading {input_file}...")
    jsonl_data = load_jsonl(input_file)
    print(f"  Loaded {len(jsonl_data)} entries")
    
    # Convert to merged format
    print("\nConverting to merged format...")
    merged_data = []
    for entry in jsonl_data:
        converted = convert_jsonl_contents_to_merged(entry)
        if converted:
            merged_data.append(converted)
    print(f"  Converted {len(merged_data)} entries")
    
    # Merge with existing merged file
    existing_merged = load_json_array(base_dir / "arabizi_init_data_input_output.json")
    all_merged = existing_merged + merged_data
    save_json(base_dir / "arabizi_init_data_input_output.json", all_merged)
    print(f"  Merged: {len(existing_merged)} existing + {len(merged_data)} new = {len(all_merged)} total")
    
    # Convert to ShareGPT format
    print("\nConverting to ShareGPT format...")
    sharegpt_data = []
    for entry in merged_data:
        converted = convert_merged_to_sharegpt(entry)
        if converted:
            sharegpt_data.append(converted)
    
    existing_sharegpt = load_json_array(base_dir / "arabizi_init_data_sharegpt.json")
    all_sharegpt = existing_sharegpt + sharegpt_data
    save_json(base_dir / "arabizi_init_data_sharegpt.json", all_sharegpt)
    print(f"  Merged: {len(existing_sharegpt)} existing + {len(sharegpt_data)} new = {len(all_sharegpt)} total")
    
    # Convert to ShareGPTGemma format
    print("\nConverting to ShareGPTGemma format...")
    sharegptgemma_data = []
    for entry in merged_data:
        converted = convert_merged_to_sharegptgemma(entry)
        if converted:
            sharegptgemma_data.append(converted)
    
    existing_sharegptgemma = load_json_array(base_dir / "arabizi_init_data_sharegptgemma.json")
    all_sharegptgemma = existing_sharegptgemma + sharegptgemma_data
    save_json(base_dir / "arabizi_init_data_sharegptgemma.json", all_sharegptgemma)
    print(f"  Merged: {len(existing_sharegptgemma)} existing + {len(sharegptgemma_data)} new = {len(all_sharegptgemma)} total")
    
    # Convert to JSONL contents format
    print("\nConverting to JSONL contents format...")
    jsonl_contents_data = []
    for entry in merged_data:
        converted = convert_merged_to_jsonl_contents(entry)
        if converted:
            jsonl_contents_data.append(converted)
    
    # Load existing JSONL
    existing_jsonl_contents = load_jsonl(base_dir / "arabizi_init_data_gemini.jsonl")
    all_jsonl_contents = existing_jsonl_contents + jsonl_contents_data
    save_jsonl(base_dir / "arabizi_init_data_gemini.jsonl", all_jsonl_contents)
    print(f"  Merged: {len(existing_jsonl_contents)} existing + {len(jsonl_contents_data)} new = {len(all_jsonl_contents)} total")
    
    # Convert to Cohere Aya format
    print("\nConverting to Cohere Aya format...")
    cohere_aya_data = []
    for entry in merged_data:
        converted = convert_merged_to_cohere_aya(entry)
        if converted:
            cohere_aya_data.append(converted)
    
    existing_cohere = load_json_array(base_dir / "arabizi_init_data_cohere_aya.json")
    all_cohere = existing_cohere + cohere_aya_data
    save_json(base_dir / "arabizi_init_data_cohere_aya.json", all_cohere)
    print(f"  Merged: {len(existing_cohere)} existing + {len(cohere_aya_data)} new = {len(all_cohere)} total")
    
    print(f"\n✓ Successfully converted and merged {len(merged_data)} entries from {input_file}")
    print(f"  Updated all format files in {base_dir}")


if __name__ == "__main__":
    main()

