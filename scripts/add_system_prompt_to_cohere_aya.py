import json
import sys
from pathlib import Path


def add_system_prompt_to_cohere_aya(input_path: Path, output_path: Path):
    """
    Add system_prompt field to each entry in Cohere Aya format JSON.
    """
    # System prompt text
    system_prompt = (
        "You are a Lebanese Internet Language (Lebanese Arabizi) agent.\n\n"
        "Your role is to write and respond naturally in *Lebanese Arabizi*, "
        "the way Lebanese people chat online using Latin letters and numbers (3, 7, 2, etc.)."
    )
    
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
    
    # Add system_prompt to each entry
    updated_entries = []
    for entry in entries:
        updated_entry = {
            "system_prompt": system_prompt,
            **entry  # Preserve existing fields (inputs, targets)
        }
        updated_entries.append(updated_entry)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated_entries, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"Added system_prompt to {len(updated_entries)} entries")
    print(f"Output written to {output_path}")
    
    return updated_entries


def main():
    """Main function to add system_prompt to Cohere Aya JSON."""
    base_dir = Path("data/fine_tuning")
    
    # Input and output files
    input_file = base_dir / "arabizi_init_data_cohere_aya.json"
    output_file = base_dir / "arabizi_init_data_cohere_aya.json"  # Overwrite same file
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading {input_file}...")
    add_system_prompt_to_cohere_aya(input_file, output_file)
    print(f"\n✓ Successfully added system_prompt to all entries in {output_file}")


if __name__ == "__main__":
    main()

