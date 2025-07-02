from main import WuPreTrainer, CantoPreTrainer
from argparse import ArgumentParser

def run(args):
    if args.lang == "yue":
        model = CantoPreTrainer()
        model.train()
    elif args.lang == "wuu":
        model = WuPreTrainer()
        model.train()
    else:
        print(f"{args.lang} is not supported. Please choose from: yue, wuu")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue")
    args = parser.parse_args()
    run(args)
