from main import WuPreTrainer, CantoPreTrainer, CantoNLIFineTuner, CantoPOSFineTuner
from argparse import ArgumentParser

def run(args):
    if args.pretrain:
        if args.lang == "yue":
            model = CantoPreTrainer(model_dir=args.model_dir)
            model.train()
        elif args.lang == "wuu":
            model = WuPreTrainer(model_dir=args.model_dir)
            model.train()
        else:
            print(f"{args.lang} pre-training is not supported. Please choose from: yue, wuu")
    if args.finetune:
        if args.lang == "yue":
            if args.task == "nli":
                model = CantoNLIFineTuner(args.lang, model_dir=args.model_dir)
                model.finetune()
            if args.task == "pos":
                model = CantoPOSFineTuner(args.lang, model_dir=args.model_dir)
                model.finetune()
            else:
                print(f"{args.task} fine-tuning is not supported. Please choose from: pos, nli")
        else:
            print(f"{args.lang} fine-tuning is not supported. Please choose from: yue")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue")
    parser.add_argument("--model_dir", default="./models/bert-base-chinese-local")
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--task", type=str, default="")
    args = parser.parse_args()

    if args.task:
        args.finetune = True
    run(args)
