import argparse
import os

from shutil import copyfile
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-folder", type=str, required=True)
    parser.add_argument("--out-folder", type=str, required=True)
    return parser


def main(args):
    os.makedirs(args.out_folder, exist_ok=True)
    filenames = sorted(os.listdir(args.in_folder))

    for i, filename_in in enumerate(tqdm(filenames)):
        ext = filename_in.split(".")[-1]
        path_in = os.path.join(args.in_folder, filename_in)

        filename_out = "{:0>6}".format(i) + "." + ext
        path_out = os.path.join(args.out_folder, filename_out)

        copyfile(path_in, path_out)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
