from datasets import load_from_disk
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
