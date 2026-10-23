"""
Convert Faknow Politifact JSON data to the format expected by politifact_loader.py
"""

import json
import pandas as pd
import os

def convert_json_to_csv(input_file, output_file):
    """Convert Faknow JSON to CSV format with required columns."""
    print(f"Converting {input_file} to {output_file}...")

    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame with required columns
    df_data = []
    for idx, item in enumerate(data):
        df_data.append({
            'id': idx,
            'text': item['text'],
            'label': item['label'],
            'graph_data': None  # No graph data available in faknow dataset
        })

    df = pd.DataFrame(df_data)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[OK] Converted {len(df)} samples")
    print(f"  - Real news (label=0): {len(df[df['label']==0])}")
    print(f"  - Fake news (label=1): {len(df[df['label']==1])}")
    return df

def main():
    # Source directory (faknow)
    source_dir = '../faknow'

    # Destination directory (fyp_code/dataset)
    dest_dir = './dataset'
    os.makedirs(dest_dir, exist_ok=True)

    # Convert all files
    print("="*60)
    print("Converting Politifact (Faknow) dataset to required format")
    print("="*60)
    print()

    train_df = convert_json_to_csv(
        os.path.join(source_dir, 'train.json'),
        os.path.join(dest_dir, 'train.csv')
    )
    print()

    val_df = convert_json_to_csv(
        os.path.join(source_dir, 'valid.json'),
        os.path.join(dest_dir, 'val.csv')
    )
    print()

    test_df = convert_json_to_csv(
        os.path.join(source_dir, 'test.json'),
        os.path.join(dest_dir, 'test.csv')
    )
    print()

    print("="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print()
    print("Dataset statistics:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    print()
    print("Files saved to:", dest_dir)

if __name__ == '__main__':
    main()
