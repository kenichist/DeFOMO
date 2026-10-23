import json
import sys
from typing import List, Any, Dict


def parse_multi_json(text: str) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    depth = 0
    buf = []
    in_str = False
    escape = False

    for ch in text:
        buf.append(ch)

        if escape:
            escape = False
            continue

        if ch == '\\':
            escape = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if in_str:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            # When depth returns to 0, we've closed a top-level object
            if depth == 0 and buf:
                obj_str = ''.join(buf).strip()
                if obj_str:
                    try:
                        objs.append(json.loads(obj_str))
                    except json.JSONDecodeError as e:
                        # You can print/log this if you want to debug bad objects
                        # print(f"Failed to parse object: {e}")
                        pass
                buf = []

    return objs


def load_documents(path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a file that is either:
      - A valid JSON array: [ {...}, {...}, ... ]
      - Or multiple concatenated JSON objects: { ... }\n{ ... }\n...

    Returns a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Check if file is a proper JSON array
    first_non_ws = next((ch for ch in text if not ch.isspace()), "")
    if first_non_ws == "[":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("JSON top-level is not a list/array.")
    else:
        # Fall back to parsing concatenated objects
        return parse_multi_json(text)


def filter_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only documents where:
      Non_news_domain == False
      Lang == "N/A"
      Country == "N/A"
    """
    filtered = []
    for doc in docs:
        non_news = doc.get("Non_news_domain", True)

        if (non_news is False):
            filtered.append(doc)

    return filtered


def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_mediarank_unknown.py input.json output.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    docs = load_documents(input_path)
    filtered = filter_documents(docs)

    # Write out as a pretty JSON array
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(docs)} documents.")
    print(f"Kept   {len(filtered)} documents with Non_news_domain == false "
          f"AND Lang == 'N/A' AND Country == 'N/A'.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
