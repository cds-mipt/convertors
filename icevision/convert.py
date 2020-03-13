import argparse
import json
import os

from tqdm import tqdm
from PIL import Image

from utils import load_annot_as_df
from categories import get_category_name


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-folder", type=str, required=True)
    parser.add_argument("--ext", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--categories", type=str, required=True)
    parser.add_argument("--coco-gt-out", type=str, required=True)
    return parser


def main(args):
    with open(args.categories, "r") as f:
        categories = json.load(f)["categories"]
    name_to_cat_id = {cat["name"]: cat["id"] for cat in categories}

    names = sorted(filename.split(".")[0] for filename in os.listdir(args.csv_folder))

    # готовим поле images
    images, name_to_image_id = [], {}
    for name in tqdm(names):
        image_id = int(name)
        filename = name + "." + args.ext
        path = os.path.join(args.images, filename)

        name_to_image_id[name] = image_id

        w, h = Image.open(path).size
        image = dict(file_name=filename, id=image_id, width=w, height=h)
        images.append(image)

    annotations = []
    for name in tqdm(names):
        filename = name + ".tsv"
        path = os.path.join(args.csv_folder, filename)

        df = load_annot_as_df(path)
        for r in df.to_dict("records"):
            cat_name = get_category_name(
                cls=r["class"],
                temporary=r["temporary"],
                data=r["data"]
            )
            x, y, w, h = r["xtl"], r["ytl"], r["xbr"] - r["xtl"], r["ybr"] - r["ytl"]
            x, y, w, h = x / 2, y / 2, w / 2, h / 2  # из за конвертации в JPEG
            annotation = dict(
                id=len(annotations) + 1,
                bbox=[x, y, w, h],
                segmentation=[[x, y, x, y + h, x + w, y + h, x + w, y]],
                iscrowd=0,
                image_id=name_to_image_id[name],
                category_id=name_to_cat_id[cat_name],
                area=w * h
            )
            annotations.append(annotation)

    coco_gt = dict(images=images, categories=categories, annotations=annotations)
    with open(args.coco_gt_out, "w") as f:
        json.dump(coco_gt, f)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
