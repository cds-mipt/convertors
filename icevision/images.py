import argparse
import os
import json

from PIL import Image
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--categories", type=str, required=True)
    parser.add_argument("--coco-gt-out", type=str, required=True)
    return parser


def main(args):
    with open(args.categories, "r") as f:
        categories = json.load(f)["categories"]

    images = []
    for filename in tqdm(os.listdir(args.images)):
        name = filename.split(".")[0]
        image_id = int(name)
        path = os.path.join(args.images, filename)
        w, h = Image.open(path).size
        image = dict(file_name=filename, id=image_id, width=w, height=h)
        images.append(image)

    coco_gt_out = dict(images=images, categories=categories)
    with open(args.coco_gt_out, "w") as f:
        json.dump(coco_gt_out, f)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
