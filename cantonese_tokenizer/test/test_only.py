import sentencepiece as spm

def test_tokenizer():
    sp = spm.SentencePieceProcessor(model_file="cantonese_sp.model")

    test_sent = "我今日食咗飯未？"
    pieces = sp.encode(test_sent, out_type=str)
    ids = sp.encode(test_sent, out_type=int)
    decoded = sp.decode(ids)

    print("Test sentence:", test_sent)
    print("Pieces:", pieces)
    print("IDs:", ids)
    print("Decoded:", decoded)

if __name__ == "__main__":
    test_tokenizer()
