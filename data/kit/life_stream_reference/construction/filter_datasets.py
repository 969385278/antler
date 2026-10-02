"""
Filter the wiki and character datasets by removing excluded questions.

Excluded questions fall into two categories:
  - "stale": questions the base model (LLaMA 3.1-8B-Instruct) can already answer
    without any fine-tuning (pre-existing knowledge)
  - "irag_n1_fail": questions where iRAG retrieval at N=1 fails (question quality
    issue, not a model limitation)

Each QA pair in the datasets already carries boolean flags (stale, irag_n1_fail).
This script removes QA pairs where either flag is True.

Usage:
    python filter_datasets.py \
        --wiki_in ../path/to/wiki_dataset_2026_1000_with_qa_paraphrases.json \
        --char_in ../path/to/elias_zhang_dataset_with_paraphrases.json \
        --wiki_out wiki_dataset_filtered.json \
        --char_out character_dataset_filtered.json
"""

import json
import argparse
import copy


def filter_wiki(dataset):
    filtered = []
    total_qa = 0
    kept_qa = 0
    for entry in dataset:
        new_entry = copy.deepcopy(entry)
        original_qa = entry["qa"]
        tags = entry.get("qa_tags", [{}] * len(original_qa))
        kept = []
        for qa_pair, tag in zip(original_qa, tags):
            total_qa += 1
            if not tag.get("stale", False) and not tag.get("irag_n1_fail", False):
                kept.append(qa_pair)
                kept_qa += 1
        new_entry["qa"] = kept
        # Remove qa_tags from filtered output (flags already applied)
        new_entry.pop("qa_tags", None)
        filtered.append(new_entry)
    return filtered, total_qa, kept_qa


def filter_character(stories):
    filtered = []
    total_qa = 0
    kept_qa = 0
    for entry in stories:
        new_entry = copy.deepcopy(entry)
        original_qa = entry["qa_pairs"]
        kept = []
        for qa_pair in original_qa:
            total_qa += 1
            if not qa_pair.get("stale", False) and not qa_pair.get("irag_n1_fail", False):
                clean_pair = {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                }
                kept.append(clean_pair)
                kept_qa += 1
        new_entry["qa_pairs"] = kept
        filtered.append(new_entry)
    return filtered, total_qa, kept_qa


def main():
    parser = argparse.ArgumentParser(
        description="Filter datasets by removing excluded questions"
    )
    parser.add_argument("--wiki_in", required=True)
    parser.add_argument("--char_in", required=True)
    parser.add_argument("--wiki_out", default="wiki_dataset_filtered.json")
    parser.add_argument("--char_out", default="character_dataset_filtered.json")
    args = parser.parse_args()

    # Filter wiki
    with open(args.wiki_in) as f:
        wiki_data = json.load(f)
    wiki_filtered, wiki_total, wiki_kept = filter_wiki(wiki_data)
    print(f"Wiki: {wiki_total} -> {wiki_kept} QA pairs "
          f"({wiki_total - wiki_kept} excluded)")
    with open(args.wiki_out, "w") as f:
        json.dump(wiki_filtered, f, indent=2)
    print(f"  Saved to {args.wiki_out}")

    # Filter character
    with open(args.char_in) as f:
        char_data = json.load(f)
    char_filtered, char_total, char_kept = filter_character(char_data["stories"])
    print(f"Character: {char_total} -> {char_kept} QA pairs "
          f"({char_total - char_kept} excluded)")
    char_out = {"metadata": char_data["metadata"], "stories": char_filtered}
    with open(args.char_out, "w") as f:
        json.dump(char_out, f, indent=2)
    print(f"  Saved to {args.char_out}")


if __name__ == "__main__":
    main()
