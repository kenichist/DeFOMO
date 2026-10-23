# map_id.py
import json
from json import JSONDecoder

def load_multiple_json_objects(path: str):
    """Load multiple JSON objects concatenated in a single file."""
    decoder = JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    objs = []
    idx = 0
    n = len(text)

    while idx < n:
        # skip whitespace between objects
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break

        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end  # continue from where this object ended

    return objs


# --- use it on your news-source file ---
SOURCE_FILE = "NewsSource_property.json"   # <-- change to your actual filename

rows = load_multiple_json_objects(SOURCE_FILE)

# Build a mapping: public_suffix_id -> domain (or whole row)
id_to_suffix = {
    row["public_suffix_id"]: row["public_suffix"]
    for row in rows
    if "public_suffix_id" in row
}

id_to_source = {
    row["public_suffix_id"]: row
    for row in rows
    if "public_suffix_id" in row
}

# Example: what does suffix_id 11 map to?
sid = 11
print("suffix_id:", sid)
print("domain:", id_to_suffix.get(sid))         # should print: cnn.com
print("short title:", id_to_source[sid].get("short_title"))
