import re
from collections import Counter
from transformers import BertTokenizer

try:
    from opencc import OpenCC
    t2s = OpenCC('t2s')
    s2t = OpenCC('s2t')
except Exception:
    t2s = s2t = None

# ------------------------
# Load cleaned vocab tokens
# ------------------------
with open("analysis_outputs/clean_vocab.txt", encoding="utf-8") as f:
    tokens = [line.strip() for line in f if line.strip()]

vsz = len(tokens)

print("=== Using CLEANED VOCAB ===")
print(f"Total tokens: {vsz}")

def is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

def classify_ts(piece: str):
    """Check if a token is traditional/simplified/unchanged"""
    if not t2s or not s2t:
        return (False, False, True)
    p = piece.replace('▁', '')
    if not p:
        return (False, False, True)
    to_s = t2s.convert(p)
    to_t = s2t.convert(p)
    has_trad = (p != to_s)
    has_simp = (p != to_t)
    unchanged = (not has_trad and not has_simp)
    return (has_trad, has_simp, unchanged)

# Count 2-char tokens
two_char_tokens = [tok for tok in tokens if len(tok) == 2 and all(is_cjk(c) for c in tok)]

# TS classification
trad_tokens, simp_tokens, neutral_tokens = [], [], []
for piece in tokens:
    ht, hs, unch = classify_ts(piece)
    if ht: trad_tokens.append(piece)
    if hs: simp_tokens.append(piece)
    if unch: neutral_tokens.append(piece)

# Cantonese-specific coverage
canto_chars = set("咗嘢佢啲冇嚟哋嗰乜嘅喺唔啱喎噉咁")
canto_token_hits = []
char_counter = Counter()
for piece in tokens:
    if any(ch in canto_chars for ch in piece):
        canto_token_hits.append(piece)
        for ch in piece:
            if ch in canto_chars:
                char_counter[ch] += 1

print("=== Cleaned Vocabulary Analysis ===")
print(f"# 2-char CJK tokens: {len(two_char_tokens)}")
print("Traditional-only tokens:", len(trad_tokens))
print("Simplified-only tokens :", len(simp_tokens))
print("Unchanged tokens       :", len(neutral_tokens))
print(f"Cantonese-specific token hits: {len(canto_token_hits)}")
print("Top Cantonese characters:", char_counter.most_common(15))
print()

# Baseline comparison
print("=== Hugging Face Baselines ===")
tok_cn = BertTokenizer.from_pretrained("bert-base-chinese")
tok_canto = BertTokenizer.from_pretrained("hon9kon9ize/bert-base-cantonese")

def coverage(tokenizer, name):
    covered = {c: tokenizer.tokenize(c) != ["[UNK]"] for c in canto_chars}
    total = sum(covered.values())
    print(f"{name}: {total}/{len(canto_chars)} covered -> {covered}")

coverage(tok_cn, "bert-base-chinese")
coverage(tok_canto, "bert-base-cantonese")

# Compare with our cleaned vocab
mine_covered = {c: any(c in piece for piece in tokens) for c in canto_chars}
print("ours (cleaned vocab):", sum(mine_covered.values()), f"/{len(canto_chars)} covered -> {mine_covered}")
