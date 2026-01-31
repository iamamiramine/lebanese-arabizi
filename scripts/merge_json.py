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


def load_json_array(path: Path):
    """Load JSON array or object from file."""
    text = path.read_text(encoding="utf-8")
    text_stripped = text.strip()
    # Try direct parse first
    try:
        data = json.loads(text_stripped)
        if isinstance(data, list):
            return data
        # If it's a single object, wrap into list
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Attempt to recover sequences of objects without surrounding brackets
    # e.g., { ... }, { ... }, { ... }
    try:
        candidate = f"[{text_stripped.strip().rstrip(',')}]"
        data = json.loads(candidate)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from {path}: {e}")


def main(argv):
    if len(argv) < 2:
        print(
            "Usage: python scripts/merge_json.py <input1> <input2> ... [--out <output>]",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = Path("data/books/merged.json")
    inputs = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            if i + 1 >= len(argv):
                print("--out requires a path", file=sys.stderr)
                sys.exit(2)
            out_path = Path(argv[i + 1])
            i += 2
        else:
            inputs.append(Path(argv[i]))
            i += 1

    merged = []
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        
        # Check if file is JSONL format
        if p.suffix.lower() == ".jsonl":
            data = load_jsonl(p)
        else:
            data = load_json_array(p)
        
        merged.extend(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Merged {len(inputs)} files -> {out_path} with {len(merged)} items")


if __name__ == "__main__":
    main(sys.argv[1:])


