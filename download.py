from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM
from argparse import ArgumentParser

def download(args):
    print("Downloading datasets and models...")
    if args.lang == "yue":
        print("Downloading Cantonese Wiki dataset...")
        ds = load_dataset("R5dwMg/zh-wiki-yue-long")
        ds.save_to_disk("./yue-wiki-local")
        print("Downloading Cantonese NLI dataset...")
        ds = load_dataset("hon9kon9ize/yue-all-nli")
        ds.save_to_disk("./yue-nli-local")
        print("Downloading BERT tokenizer and model...")
        BertTokenizerFast.from_pretrained('bert-base-chinese').save_pretrained(args.model_dir)
    elif args.lang == "wuu":
        print("Downloading Wu Wiki dataset...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.wuu")
        ds.save_to_disk("./wuu-wiki-local")
        print("Downloading BERT tokenizer...")
        BertTokenizerFast.from_pretrained('bert-base-chinese').save_pretrained(args.model_dir)
    else:
        print(f"{args.lang} is not supported. Please choose from: yue, wuu")
        return
    print("Downloading BERT model...")
    BertForMaskedLM.from_pretrained('bert-base-chinese').save_pretrained(args.model_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue", choices=["yue", "wuu"],
                        help="Language to download data for")
    parser.add_argument("--model_dir", default="./bert-base-chinese-local",
                        help="Directory to save the model and tokenizer")
    args = parser.parse_args()
    download(args)
