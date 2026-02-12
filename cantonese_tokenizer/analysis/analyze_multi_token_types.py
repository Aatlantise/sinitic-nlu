import re
from pathlib import Path
from opencc import OpenCC

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
opencc_t2s = OpenCC("t2s")

tokens_path = Path("analysis_outputs/all_multi_character_tokens.txt")
tokens = [line.strip() for line in tokens_path.open(encoding="utf-8") if line.strip()]

# Cantonese-specific characters
canto_specific_chars = "嘅咗冇唔啱喺嗰啩喎佢乜嘢啲啦吓呀囉嘛咁噉嚟嗌嬲攰冚曱甴鳩屙"

canto_like, shared_like = [], []

def is_cantonese_word(tok: str):
    # If it contains a known Cantonese character then we take it as Cantonese
    if any(ch in tok for ch in canto_specific_chars):
        return True
    # If its simplified version differs still probably Traditional (shared)
    simp = opencc_t2s.convert(tok)
    if simp != tok:
        return False
    # Default → shared/neutral
    return False

# --- Categorize tokens ---
for t in tokens:
    if is_cantonese_word(t):
        canto_like.append(t)
    else:
        shared_like.append(t)

# --- Save results ---
out_dir = Path("analysis_outputs")
out_dir.mkdir(exist_ok=True)
output_path = out_dir / "multi_token_categorization.tsv"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("token\tcategory\n")
    for t in canto_like:
        f.write(f"{t}\tCantonese\n")
    for t in shared_like:
        f.write(f"{t}\tShared/Neutral\n")

print(f" Found {len(canto_like)} likely Cantonese-specific tokens.")
print(f" Found {len(shared_like)} shared/neutral tokens.")
print(f" Results saved to {output_path}")
