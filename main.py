# from model.torch_model import run_fn
from model.exampleModel import run_fn
import argparse
from utils.utils import *


def parse_args():
    desc = "Pytorch implementation of DeepNetwork"
    parser = argparse.ArgumentParser(description=desc)

    # dataset
    parser.add_argument('--dataset', type=str, default='test', help="The name of the dataset")

    # training
    # => Plan : To make config loader
    # parser.add_argument('--model', type=str, default='vggnet-19.cfg', help='The name of the model(config format)')

    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=28, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--feature_size', type=int, default=64, help='The size of feature size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # model name
    parser.add_argument("--model_name", type=str, default='myMnistNet', help="The name of the model")

    # network settings
    parser.add_argument('--config_dir', type=str, default='./cfg')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""


def main():
    args = vars(parse_args())

    # run
    run_fn(args=args)


if __name__ == '__main__':
    main()
