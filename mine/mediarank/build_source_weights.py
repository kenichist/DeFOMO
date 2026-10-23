import json
from typing import Dict, Tuple, Any, Iterable

# --------- Config ---------
OUTPUT_JSON = "output.json"                  # MediaRank scores with public_suffix_id
NEWS_JSON = "NewsSource_property.json"       # Mapping from public_suffix_id -> public_suffix (domain)
OUT_DOMAIN_WEIGHTS = "domain_weights.json"   # Final mapping: domain -> weight
DEFAULT_WEIGHT = 5.0                         # Fallback when a suffix_id has no score


# --------- JSON Loader (robust to many formats) ---------
def load_json_stream(path: str) -> Iterable[Any]:
    """
    Robust loader that yields JSON objects from a file that may be:
    - a single JSON array: [ {...}, {...}, ... ]
    - a single JSON object: {...}
    - newline-delimited JSON: one object per line
    - multiple arrays/objects concatenated: [..][..]{..}{..}

    It yields dicts (and may occasionally yield lists, which we handle upstream).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 1) Try parsing the whole file in one go
    try:
        data = json.loads(text)
        # If it's a list, yield each element
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            # Single dict or other
            yield data
        return
    except json.JSONDecodeError:
        pass

    # 2) Try "JSON per line"
    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                line_count += 1
                yield obj
            except json.JSONDecodeError:
                # Not line-delimited; we'll try the concatenated fallback
                line_count = 0
                break

    if line_count > 0:
        # We successfully parsed as line-delimited JSON
        return

    # 3) Fallback: concatenated JSON objects/arrays
    buf = ""
    with open(path, "r", encoding="utf-8") as f:
        for ch in f.read():
            buf += ch
            try:
                obj = json.loads(buf)
                # Parsed something valid; yield it and reset buffer
                yield obj
                buf = ""
            except json.JSONDecodeError:
                # Keep accumulating
                continue
    # If anything remains in buf that still can't be parsed, we just ignore it.


# --------- Loaders for ID→Score and ID→Domain ---------
def load_id_to_score(
    path: str,
    id_key: str = "public_suffix_id",
    score_key: str = "mediarank_score",
) -> Dict[int, float]:
    """
    Reads output.json and builds:
        public_suffix_id (int) → mediarank_score (float)
    Handles nested lists/dicts from load_json_stream.
    """
    id_to_score: Dict[int, float] = {}
    total_records = 0

    for item in load_json_stream(path):
        # Flatten if we get a list of objects
        if isinstance(item, list):
            objects = item
        else:
            objects = [item]

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            total_records += 1

            raw_sid = obj.get(id_key)
            score = obj.get(score_key)

            if raw_sid is None or score is None:
                continue

            try:
                sid = int(raw_sid)
                score_val = float(score)
            except (ValueError, TypeError):
                continue

            id_to_score[sid] = score_val

    print(
        f"[load_id_to_score] Parsed {total_records} records from {path} "
        f"→ {len(id_to_score)} unique suffix_ids with scores"
    )
    return id_to_score


def load_id_to_domain(
    path: str,
    id_key: str = "public_suffix_id",
    domain_key: str = "public_suffix",
    skip_non_news: bool = False,
) -> Dict[int, str]:
    """
    Reads NewsSource_property.json and builds:
        public_suffix_id (int) → public_suffix (domain, e.g. 'cnn.com')

    If skip_non_news is True, skips entries with Non_news_domain == True.
    """
    id_to_domain: Dict[int, str] = {}
    total_records = 0

    for item in load_json_stream(path):
        # Flatten if we get a list of objects
        if isinstance(item, list):
            objects = item
        else:
            objects = [item]

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            total_records += 1

            if skip_non_news and obj.get("Non_news_domain") is True:
                continue

            raw_sid = obj.get(id_key)
            domain = obj.get(domain_key)

            if raw_sid is None or not domain:
                continue

            try:
                sid = int(raw_sid)
            except (ValueError, TypeError):
                continue

            id_to_domain[sid] = str(domain).strip()

    print(
        f"[load_id_to_domain] Parsed {total_records} records from {path} "
        f"→ {len(id_to_domain)} id→domain mappings (skip_non_news={skip_non_news})"
    )
    return id_to_domain


# --------- Build domain→weight ---------
def build_domain_weights(
    id_to_score: Dict[int, float],
    id_to_domain: Dict[int, str],
    default_weight: float = DEFAULT_WEIGHT,
) -> Tuple[Dict[str, float], int]:
    """
    Combines:
      - id_to_score:  suffix_id → score
      - id_to_domain: suffix_id → domain (e.g., 'cnn.com')

    Returns:
      - domain_to_weight: domain → weight
      - default_count: how many domains fell back to default_weight
    """
    domain_to_weight: Dict[str, float] = {}
    default_count = 0

    for sid, domain in id_to_domain.items():
        score = id_to_score.get(sid)
        if score is None:
            weight = default_weight
            default_count += 1
        else:
            weight = float(score)
        domain_to_weight[domain] = weight

    print(f"[build_domain_weights] Total domains: {len(id_to_domain)}")
    print(
        f"[build_domain_weights] Domains using default {default_weight}: {default_count}"
    )
    return domain_to_weight, default_count


def save_domain_weights(domain_to_weight: Dict[str, float], path: str) -> None:
    """
    Saves domain_to_weight as a JSON object:
      { "cnn.com": 57.0, "nytimes.com": 100.0, ... }
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            domain_to_weight,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(f"[save_domain_weights] Saved domain weights to {path}")


# --------- Debug helpers ---------
def debug_known_domains(
    domains: Iterable[str],
    id_to_domain: Dict[int, str],
    id_to_score: Dict[int, float],
    domain_to_weight: Dict[str, float],
) -> None:
    """Print info for some well-known domains (e.g. washingtonpost, cnn, nytimes)."""
    for dom in domains:
        sid = None
        for k, v in id_to_domain.items():
            if v == dom:
                sid = k
                break

        if sid is not None:
            print(f"\n[debug] Domain: {dom}")
            print(f"  public_suffix_id: {sid}")
            print(f"  id_to_score[{sid}] = {id_to_score.get(sid)}")
            print(f"  domain_to_weight['{dom}'] = {domain_to_weight.get(dom)}")
        else:
            print(f"\n[debug] Domain: {dom} not found in NewsSource_property.json")


def debug_default_domains(
    id_to_domain: Dict[int, str],
    id_to_score: Dict[int, float],
    max_examples: int = 20,
) -> None:
    """Print a sample of domains that still ended up with default weight (no score)."""
    print("\n[debug] Sample domains using default score:")
    shown = 0
    for sid, domain in id_to_domain.items():
        if sid not in id_to_score:
            print(f"  sid={sid:<6d}  domain={domain}")
            shown += 1
            if shown >= max_examples:
                break
    if shown == 0:
        print("  (None – every suffix_id has a score.)")


# --------- Main ---------
def main():
    # 1) Load mappings
    id_to_score = load_id_to_score(OUTPUT_JSON)
    id_to_domain = load_id_to_domain(NEWS_JSON, skip_non_news=False)

    # 2) Build domain → weight mapping
    domain_to_weight, default_count = build_domain_weights(
        id_to_score, id_to_domain, DEFAULT_WEIGHT
    )

    # 3) Debug for some specific big outlets
    debug_known_domains(
        ["washingtonpost.com", "cnn.com", "nytimes.com"],
        id_to_domain,
        id_to_score,
        domain_to_weight,
    )

    # 4) Show a few domains that are still using the default
    if default_count > 0:
        debug_default_domains(id_to_domain, id_to_score, max_examples=20)

    # 5) Save final mapping
    save_domain_weights(domain_to_weight, OUT_DOMAIN_WEIGHTS)


if __name__ == "__main__":
    main()
