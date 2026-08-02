import json
import os
from collections import Counter

def analyze(path):
    counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            counter[obj["label"]] += 1

    total = counter[0] + counter[1]

    print(f"\n{path}")
    print(f"Total: {total}")
    print(f"acceptable (1): {counter[1]} ({counter[1]/total:.2%})")
    print(f"unacceptable (0): {counter[0]} ({counter[0]/total:.2%})")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze(os.path.join(script_dir, "finetune_train.jsonl"))
    analyze(os.path.join(script_dir, "finetune_test.jsonl"))
