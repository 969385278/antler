"""
Build wiki_dataset_2026_1000_long.json — 1000 articles with mean ~103 tokens
to match character dataset length.

Pipeline:
1. Load pre-filtered candidates (wiki_candidates_90_125.json, [90-125] raw tokens, deduped)
2. GPT-clean each article via Harvard API (concurrent, 8 workers)
3. Filter out articles with ":" in cleaned_text
4. Random sample 1000 (seed=42)
5. Save final dataset
"""

import json
import requests
import time
import random
import os
import concurrent.futures
from threading import Lock
from datetime import datetime

API_KEY = os.getenv("HARVARD_API_KEY", "")
API_URL = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions"

def api_clean_text(text):
    prompt = (
        "If the following text has format or template issues "
        "(e.g. excessive use of punctuations, symbols, template tags), "
        "please clean it up. Only return the cleaned text, no other text or comments. "
        "If the text is already clean, just return it as is: \n\n " + text
    )
    payload = json.dumps({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1
    })
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    for attempt in range(20):
        try:
            response = requests.post(API_URL, headers=headers, data=payload, timeout=60)
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(5)
    return None


def main():
    with open("wiki_candidates_90_125.json") as f:
        candidates = json.load(f)
    print(f"[{datetime.now():%H:%M:%S}] Loaded {len(candidates)} candidates", flush=True)

    # Check for partial progress
    output_path = "wiki_candidates_90_125_cleaned.json"
    cleaned = []
    done_titles = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            cleaned = json.load(f)
        done_titles = {item["title"] for item in cleaned}
        print(f"[{datetime.now():%H:%M:%S}] Resuming — {len(cleaned)} already cleaned", flush=True)

    remaining = [c for c in candidates if c["title"] not in done_titles]
    print(f"[{datetime.now():%H:%M:%S}] {len(remaining)} articles to clean", flush=True)

    lock = Lock()
    save_freq = 25
    processed = [0]
    total = len(remaining)

    def process_one(item):
        cleaned_text = api_clean_text(item["text"])
        if cleaned_text is None:
            return None
        return {
            "title": item["title"],
            "text": item["text"],
            "cleaned_text": cleaned_text
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one, item): item for item in remaining}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                print(f"  FAILED: {futures[future]['title']}", flush=True)
                continue
            with lock:
                cleaned.append(result)
                processed[0] += 1
                if processed[0] % save_freq == 0:
                    with open(output_path, "w") as f:
                        json.dump(cleaned, f, indent=2)
                    pct = processed[0] / total * 100
                    print(f"[{datetime.now():%H:%M:%S}] Cleaned {processed[0]}/{total} ({pct:.0f}%)", flush=True)

    # Final save of all cleaned
    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"\n[{datetime.now():%H:%M:%S}] Cleaning done: {len(cleaned)} articles", flush=True)

    # Filter out articles with ":"
    no_colon = [item for item in cleaned if ":" not in item["cleaned_text"]]
    print(f"After colon filter: {len(no_colon)}", flush=True)

    # Sample 1000
    random.seed(42)
    if len(no_colon) >= 1000:
        selected = random.sample(no_colon, 1000)
    else:
        print(f"WARNING: only {len(no_colon)} articles available, using all", flush=True)
        selected = no_colon

    # Save final dataset
    final_path = "wiki_dataset_2026_1000_long.json"
    with open(final_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] Saved {len(selected)} articles to {final_path}", flush=True)

    # Print stats
    from transformers import AutoTokenizer
    import numpy as np
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    raw_toks = [len(tokenizer.encode(item["text"])) - 1 for item in selected]
    clean_toks = [len(tokenizer.encode(item["cleaned_text"])) - 1 for item in selected]
    a_r = np.array(raw_toks)
    a_c = np.array(clean_toks)
    print(f"\nFinal stats ({len(selected)} articles):")
    print(f"  Raw tokens:     mean={a_r.mean():.1f} std={a_r.std():.1f} min={a_r.min()} max={a_r.max()}")
    print(f"  Cleaned tokens: mean={a_c.mean():.1f} std={a_c.std():.1f} min={a_c.min()} max={a_c.max()}")


if __name__ == "__main__":
    main()
