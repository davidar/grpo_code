#!/usr/bin/env python3
"""
Convert synthetic conversations dataset to GRPO-compatible format.

Transforms from:
{
  "conversation_id": "synthetic_0000",
  "starter": "...", 
  "context": "<alice> message\n<bob> reply...",
  "messages": [...],
  "metadata": {...}
}

To:
{
  "text": "<alice> message\n<bob> reply...\n<Em>"
}

This creates a completion dataset where Em generates responses after the conversation context.
"""

import json
import argparse
from pathlib import Path


def convert_conversation(item):
    """Convert a single conversation item to GRPO format."""
    context = item.get("context", "")
    
    if not context:
        print(f"Warning: Empty context in {item.get('conversation_id', 'unknown')}")
        return None
    
    # Add <Em> at the end so the model learns to complete after the conversation
    # This matches Em's expected format: conversation context + <Em> + response
    text = context + "\n<Em>"
    
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic conversations to GRPO format")
    parser.add_argument(
        "--input",
        default="data/synthetic_conversations.jsonl",
        help="Input synthetic conversations file"
    )
    parser.add_argument(
        "--output", 
        default="data/em_conversations_grpo.jsonl",
        help="Output GRPO-compatible file"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    
    print(f"Converting {input_path} -> {output_path}")
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                item = json.loads(line.strip())
                converted = convert_conversation(item)
                
                if converted:
                    outfile.write(json.dumps(converted) + '\n')
                    converted_count += 1
                else:
                    skipped_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1
    
    print(f"Conversion complete!")
    print(f"  Converted: {converted_count} conversations")
    print(f"  Skipped: {skipped_count} conversations")
    print(f"  Output: {output_path}")
    
    # Show a sample of the output
    if converted_count > 0:
        print(f"\nSample output:")
        with open(output_path, 'r') as f:
            sample = json.loads(f.readline())
            # Show first 200 chars of the text
            sample_text = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
            print(f"  {sample_text}")
    
    return 0


if __name__ == "__main__":
    exit(main())
