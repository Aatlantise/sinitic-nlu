import os
from datasets import load_dataset

def prepare_corpus():
    # Download Hugging Face Cantonese Sentences
    print("Downloading Hugging Face Cantonese dataset...")
    ds = load_dataset("raptorkwok/cantonese_sentences")

    with open("cantonese_hf.txt", "w", encoding="utf-8") as f:
        for row in ds["train"]:
            f.write(row["content"].strip() + "\n")

    # Download Wikipedia Dump
    wiki_path = "cantonese_wiki.txt"
    if not os.path.exists(wiki_path):
        raise FileNotFoundError(
            "cantonese_wiki.txt not found. "
            "Please download yuewiki dump and process with WikiExtractor."
        )

    # Merge corpora（wiki + hugging face）
    with open("cantonese.txt", "w", encoding="utf-8") as out_f:
        for fname in ["cantonese_hf.txt", "cantonese_wiki.txt"]:
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    out_f.write(line)

    print("Training corpus written to cantonese.txt")

if __name__ == "__main__":
    prepare_corpus()
