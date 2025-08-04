from datasets import load_from_disk, Dataset, DatasetDict
from collections import Counter
from tqdm import tqdm
import pandas as pd
import os


def lexical_similarity(s1: str, s2: str) -> float:
    """
    Calculate the lexical similarity between two strings using Jaccard similarity,
    considering only non-Latin characters.
    :param s1: str in a Sinitic language
    :param s2: str in a Sinitic language
    :return: float between [0, 1]
    """
    s1 = ''.join(filter(lambda x: x.isalpha() or x.isspace(), s1))
    s2 = ''.join(filter(lambda x: x.isalpha() or x.isspace(), s2))
    s1_counter = Counter(list(s1))
    s2_counter = Counter(list(s2))

    intersection = sum((s1_counter & s2_counter).values())
    union = sum((s1_counter | s2_counter).values())

    if union == 0:
        return 0.0
    return intersection / union


def get_yue_nli_test_subset(n: int) -> pd.DataFrame:
    """
    Get the n-most lexically dissimilar (anchor-negative similarity minus anchor-positive similarity)
    examples from the test split of the Yue NLI dataset.
    :param n: Number of examples to return.
    :return: DataFrame
    """
    if not os.path.exists('data/yue-nli-local'):
        raise FileNotFoundError("The Yue NLI dataset is not found. Please first run `python download.py --lang=yue`.")
    ds = load_from_disk('data/yue-nli-local')['test']

    def compute_lexical_similarity(example):
        positive_similarity = lexical_similarity(example['anchor'], example['positive'])
        negative_similarity = lexical_similarity(example['anchor'], example['negative'])
        return negative_similarity - positive_similarity

    results = []
    for example in tqdm(ds):
        similarity = compute_lexical_similarity(example)
        results.append({
            'anchor': example['anchor'],
            'positive': example['positive'],
            'negative': example['negative'],
            'similarity': similarity
        })

    df = pd.DataFrame(results)
    df.sort_values(by='similarity', ascending=False, inplace=True)
    return df[:n]


def parse_conllu(file_path):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip metadata and empty lines
            if not line or line.startswith('#'):
                if not line:
                    if current_sentence:  # end of sentence
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                continue

            parts = line.split('\t')
            if len(parts) != 10:
                continue  # malformed line

            token, upos = parts[1], parts[3]
            current_sentence.append(token)
            current_labels.append(upos)

        # Catch any sentence that wasn’t followed by a blank line
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

    return sentences, labels


def build_dataset_dict(sentences, labels, train_ratio=0.9):
    total = len(sentences)
    train_end = int(total * train_ratio)

    dataset_splits = {
        "train": {
            "sentence": sentences[:train_end],
            "labels": labels[:train_end]
        },
        "test": {
            "sentence": sentences[train_end:],
            "labels": labels[train_end:]
        }
    }

    return DatasetDict({
        split: Dataset.from_dict(data)
        for split, data in dataset_splits.items()
    })


def conllu_to_pos_dataset():
    # Usage
    conllu_path = "./data/yue_hk-ud-test.conllu"  # Adjust this path
    sents, tags = parse_conllu(conllu_path)
    dataset = build_dataset_dict(sents, tags)

    # Optionally save for reuse
    dataset.save_to_disk("./data/yue-pos")
