import sentencepiece as spm
import unicodedata

def test_tokenizer():
    sp = spm.SentencePieceProcessor(model_file="cantonese_sp.model")

    test_sents = [
        "我今日食咗飯未？",
        "香港有好多好嘢食。",
        "電腦科學同人工智能發展得好快。",
        "佢哋喺圖書館溫書。",
    ]

    for sent in test_sents:
        pieces = sp.encode(sent, out_type=str)
        ids = sp.encode(sent, out_type=int)
        decoded = sp.decode(ids)

        print("=" * 50)
        print("Original :", sent)
        print("Pieces   :", pieces)
        print("IDs      :", ids)
        print("Decoded  :", decoded)
    
    # 1. Check vocab size
    print("Vocab size:", sp.get_piece_size())

    # 2. Check for 2-character tokens
    two_char_tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if len(sp.id_to_piece(i)) == 2]
    print("Number of 2-character tokens:", len(two_char_tokens))
    print("Sample:", two_char_tokens[:50])

    trad_count, simp_count = 0, 0
    trad_chars = set("體學會國廣電馬龍愛發與")  # Traditional
    simp_chars = set("体学会国广电马龙爱发与")  # Simplified

    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        for ch in piece:
            if ch in trad_chars:
                trad_count += 1
            if ch in simp_chars:
                simp_count += 1

    print("Traditional char tokens:", trad_count)
    print("Simplified char tokens:", simp_count)

if __name__ == "__main__":
    test_tokenizer()


