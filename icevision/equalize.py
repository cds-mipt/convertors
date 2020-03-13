import argparse
import os
import cv2

from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-folder", type=str, required=True)
    parser.add_argument("--out-folder", type=str, required=True)
    return parser


def main(args):
    os.makedirs(args.out_folder, exist_ok=True)
    filenames = sorted(os.listdir(args.in_folder))[:10]
    for filename in tqdm(filenames):
        path_in = os.path.join(args.in_folder, filename)
        path_out = os.path.join(args.out_folder, filename)

        img = cv2.imread(path_in)
        # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])

        cv2.imwrite(path_out, img)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
