import os
import sys
import time

import pandas as pd
import requests
from tqdm import tqdm


def main():
    # --- 1. Parse arguments ---
    if len(sys.argv) < 2:
        print("Usage: python image_downloader_new.py <tsv_path> [output_dir]")
        sys.exit(1)

    tsv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "images"

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Reading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    # --- 2. User-Agent header (your earlier idea, slightly improved) ---
    headers = {
        # You *can* put "your bot 0.1" here, but a browser-like UA is usually safer
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; FakedditDownloader/0.1)"
        # "User-Agent": "your bot 0.1"   # <- if you really want exactly that
    }

    # --- 3. Iterate over rows and download images ---
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row.get("image_url")

        # Skip rows without a proper URL
        if not isinstance(url, str) or not url.startswith("http"):
            continue

        # Choose an ID for the filename
        # Try several common id fields, fall back to row index
        img_id = (
            str(row.get("id"))
            if "id" in row and not pd.isna(row["id"])
            else str(row.get("submission_id", idx))
        )

        fname = os.path.join(out_dir, img_id + ".jpg")

        # Skip if we already downloaded this image
        if os.path.exists(fname):
            continue

        # Try up to 3 times per image
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                status = resp.status_code

                if status == 200:
                    # Save image
                    with open(fname, "wb") as f:
                        f.write(resp.content)

                    # Small delay to reduce chance of 429 (Too Many Requests)
                    time.sleep(1.0)
                    break

                elif status == 429:
                    # Rate limited: exponential backoff
                    wait = 3 * (attempt + 1)
                    print(f"\n[WARN] 429 for {url}, sleeping {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                    continue  # retry

                else:
                    # Other HTTP error → skip this URL
                    print(f"\n[WARN] Skip {url} (status {status})")
                    break

            except Exception as e:
                print(f"\n[ERROR] {url}: {e}")
                # brief pause then give up on this URL
                time.sleep(2)
                break


if __name__ == "__main__":
    main()
