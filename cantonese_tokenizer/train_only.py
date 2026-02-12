import sentencepiece as spm

def train_tokenizer():
    spm.SentencePieceTrainer.train(
        input="cantonese.txt",
        model_prefix="cantonese_sp",
        vocab_size=32000,
        character_coverage=0.9995,
        model_type="bpe",
        input_sentence_size=10000000,  
        shuffle_input_sentence=True
    )
    print(" Tokenizer trained: cantonese_sp.model & cantonese_sp.vocab saved.")

if __name__ == "__main__":
    train_tokenizer()
