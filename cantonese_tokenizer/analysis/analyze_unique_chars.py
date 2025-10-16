import re
from pathlib import Path
from transformers import AutoTokenizer
import sentencepiece as spm

CANTO_SP_MODEL = Path("cantonese_sp.model")
MANDARIN_MODEL_ID = "bert-base-chinese"

CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")

def is_cjk_token(s: str) -> bool:
    """Return True if all chars are CJK (no punctuation/latin)."""
    return bool(s) and all(CJK_RE.fullmatch(ch) for ch in s)

print("Loading tokenizers...")
sp = spm.SentencePieceProcessor(model_file=str(CANTO_SP_MODEL))
canto_vocab = [sp.IdToPiece(i).replace("▁", "") for i in range(sp.GetPieceSize())]

mand_tok = AutoTokenizer.from_pretrained(MANDARIN_MODEL_ID)
mand_vocab = list(mand_tok.get_vocab().keys())
mand_vocab_clean = {t.replace("##", "") for t in mand_vocab if is_cjk_token(t)}

canto_char_tokens = {t for t in canto_vocab if len(t) == 1 and is_cjk_token(t)}
canto_multi_tokens = {t for t in canto_vocab if len(t) > 1 and is_cjk_token(t)}

mand_char_tokens = {t for t in mand_vocab_clean if len(t) == 1}
mand_multi_tokens = {t for t in mand_vocab_clean if len(t) > 1}

unique_chars = sorted(c for c in canto_char_tokens if c not in mand_char_tokens)
unique_multis_raw = sorted(w for w in canto_multi_tokens if w not in mand_multi_tokens)

# filter out "Mandarin-only" multi tokens
# keep only tokens that include at least one non-Mandarin character
mand_single_chars = mand_char_tokens  # set of Mandarin chars

def is_truly_cantonese(token):
    """Return True if token has at least one character NOT in Mandarin BERT."""
    return any(ch not in mand_single_chars for ch in token)

unique_multis_filtered = [t for t in unique_multis_raw if is_truly_cantonese(t)]

print("\n=== Cantonese vs Mandarin Tokenizer Comparison ===")
print(f"Total Cantonese tokens: {len(canto_vocab)}")
print(f"Unique single-character Cantonese tokens: {len(unique_chars)}")
print(f"Unique multi-character Cantonese tokens (raw): {len(unique_multis_raw)}")
print(f"Unique multi-character Cantonese tokens (filtered): {len(unique_multis_filtered)}\n")

print("Sample unique characters:")
print("".join(unique_chars[:50]))

print("\nSample unique multi-character tokens (filtered):")
print(unique_multis_filtered[:50])

# save outputs
out_dir = Path("analysis_outputs")
out_dir.mkdir(exist_ok=True)

# Save unique single-character tokens
with open(out_dir / "unique_cantonese_characters.txt", "w", encoding="utf-8") as f:
    f.write("".join(unique_chars) + "\n")
    f.write("\n".join(unique_chars))

# Save raw multi-character list (before filtering)
with open(out_dir / "unique_cantonese_multi_tokens_raw.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(unique_multis_raw))

# Save filtered multi-character list (Cantonese-only)
with open(out_dir / "unique_cantonese_multi_tokens_filtered.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(unique_multis_filtered))

print(f"\n Saved results to:")
print(f"  - {out_dir/'unique_cantonese_characters.txt'}")
print(f"  - {out_dir/'unique_cantonese_multi_tokens_raw.txt'}")
print(f"  - {out_dir/'unique_cantonese_multi_tokens_filtered.txt'}")
print("\nDone!")
