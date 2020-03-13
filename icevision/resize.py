import argparse
import os
import cv2

from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-folder", type=str, required=True)
    parser.add_argument("--out-folder", type=str, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser


def main(args):
    os.makedirs(args.out_folder, exist_ok=True)
    filenames = sorted(os.listdir(args.in_folder))

    for i, filename in enumerate(tqdm(filenames)):
        path_in = os.path.join(args.in_folder, filename)
        path_out = os.path.join(args.out_folder, filename)

        img = cv2.imread(path_in)
        img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path_out, img)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
