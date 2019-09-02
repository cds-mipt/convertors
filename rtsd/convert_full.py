import argparse
import pandas as pd
import json
import os

from PIL import Image
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-gt", type=str, required=True)
    parser.add_argument("--full-frames", type=str, required=True)
    parser.add_argument("--coco-gt-out", type=str, required=True)
    return parser


def main(args):
    df = pd.read_csv(args.full_gt)

    labels = sorted(set(df["sign_class"]))
    categories = [{"id": cat_id, "name": label, "supercategory": "traffic sign"}
                  for cat_id, label in enumerate(labels, start=1)]  # start=1 !!!
    sign_class_to_id = {cat["name"]: cat["id"] for cat in categories}

    print("Load image info")
    filenames = sorted(os.listdir(args.full_frames))
    images = []
    for image_id, filename in enumerate(tqdm(filenames)):  # start=0 !!!
        width, height = Image.open(os.path.join(args.full_frames, filename)).size
        image = dict(id=image_id, width=width, height=height, file_name=filename)
        images.append(image)
    filename_to_id = {image["file_name"]: image["id"] for image in images}

    print("Convert annotations")
    annotations = []
    for ann_id, rec in enumerate(tqdm(df.to_dict("records")), start=1):  # start=1 !!!
        image_id = filename_to_id[rec["filename"]]
        category_id = sign_class_to_id[rec["sign_class"]]
        x, y, w, h = rec["x_from"], rec["y_from"], rec["width"], rec["height"]
        annotation = dict(
            id=ann_id,
            image_id=image_id,
            category_id=category_id,
            area=w * h,
            bbox=[x, y, w, h],
            iscrowd=0
        )
        annotations.append(annotation)

    coco_gt = dict(categories=categories, images=images, annotations=annotations)
    with open(args.coco_gt_out, "w") as f:
        json.dump(coco_gt, f, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
