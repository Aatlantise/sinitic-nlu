import re
import sentencepiece as spm
from collections import Counter
from transformers import BertTokenizer

try:
    from opencc import OpenCC
    t2s = OpenCC('t2s')
    s2t = OpenCC('s2t')
except Exception:
    t2s = s2t = None

sp = spm.SentencePieceProcessor(model_file="cantonese_sp.model")
vsz = sp.get_piece_size()

def is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

def classify_ts(piece: str):
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
two_char_tokens = []
for i in range(vsz):
    piece = sp.id_to_piece(i).lstrip('▁')
    if len(piece) == 2 and all(is_cjk(c) for c in piece):
        two_char_tokens.append(piece)

# TS classification with OpenCC
trad_tokens, simp_tokens, neutral_tokens = [], [], []
for i in range(vsz):
    piece = sp.id_to_piece(i)
    ht, hs, unch = classify_ts(piece)
    if ht: trad_tokens.append(piece)
    if hs: simp_tokens.append(piece)
    if unch: neutral_tokens.append(piece)

# Cantonese-specific coverage in our tokenizer
canto_chars = set("咗嘢佢啲冇嚟哋嗰乜嘅喺唔啱喎噉咁")
canto_token_hits = []
char_counter = Counter()
for i in range(vsz):
    piece = sp.id_to_piece(i).replace('▁', '')
    if any(ch in canto_chars for ch in piece):
        canto_token_hits.append(piece)
        for ch in piece:
            if ch in canto_chars:
                char_counter[ch] += 1

print("=== Your SentencePiece Tokenizer ===")
print(f"Vocab size: {vsz}")
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

# Compare with ours
mine_covered = {c: any(c in sp.id_to_piece(i) for i in range(sp.get_piece_size())) for c in canto_chars}
print("ours:", sum(mine_covered.values()), f"/{len(canto_chars)} covered -> {mine_covered}")
