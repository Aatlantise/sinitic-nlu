import sentencepiece as spm
from pathlib import Path

SP_MODEL_PATH = "cantonese_sp.model"
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)

out_dir = Path("analysis_outputs")
out_dir.mkdir(exist_ok=True)

# Extract multi-character tokens
def is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

multi_tokens = []
for i in range(sp.get_piece_size()):
    piece = sp.id_to_piece(i).replace("▁", "")
    if len(piece) >= 2 and all(is_cjk(ch) for ch in piece):
        multi_tokens.append(piece)

# Save all multi-character tokens to a file
output_path = out_dir / "all_multi_character_tokens.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(multi_tokens))

print(f" Extracted {len(multi_tokens)} multi-character tokens.")
print(f" Saved to {output_path}")
