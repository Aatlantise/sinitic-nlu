from main import WuPreTrainer, CantoPreTrainer
from argparse import ArgumentParser

def run(args):
    if args.lang == "yue":
        model = CantoPreTrainer(model_dir=args.model_dir)
        model.train()
    elif args.lang == "wuu":
        model = WuPreTrainer(model_dir=args.model_dir)
        model.train()
    else:
        print(f"{args.lang} is not supported. Please choose from: yue, wuu")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue")
    parser.add_argument("--model_dir", default="./bert-base-chinese-local")
    args = parser.parse_args()
    run(args)
