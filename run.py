import argparse


from Config.Config import Config
from Training.Trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Supply parameters to train the model.")
    parser.add_argument("-e", "--num_epochs", type=int, nargs='?', const=True, default=20,
                        help="The number of training epochs.")
    parser.add_argument("-d", "--device", type=str, nargs="?", const=True, default="cpu",
                        help="The device on which to run the model.")
    parser.add_argument("-c", "--num_classes", type=int, nargs="?", const=True, default=3,
                        help="The number of classes for the segmentation model to produce.")
    parser.add_argument("-l", "--learning_rate", type=float, nargs="?", const=True, default=1.e-3,
                        help="The initial learning rate.")
    parser.add_argument("-w", "--weight_decay", type=float, nargs="?", const=True, default=1.e-4,
                        help="The weight decay value for the training.")
    parser.add_argument("-p", "--prob", type=float, nargs="?", const=True, default=0.5,
                        help="The probability to apply augmentation to a batch of training data.")

    return parser.parse_args()

def init_config(args):
    config = Config(lr=args.learning_rate,
                    num_classes=args.num_epochs,
                    device=args.device,
                    w_decay=args.weight_decay,
                    prob=args.prob)
    return config

if __file__ == "__main__":
    cmd_args = get_args()
    config = init_config(cmd_args)
    trainer = Trainer(config)
    trainer.run()