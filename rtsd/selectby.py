import argparse
import pandas as pd
import json
import os

from collections import defaultdict
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-gt-in", type=str, required=True)

    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--csvs", type=str, default=None)

    parser.add_argument("--rm-cats", action="store_true")
    parser.add_argument("--cats", type=str, default=None)

    parser.add_argument("--coco-gt-out", type=str, required=True)
    return parser


def group(coco_dt):
    grouped = defaultdict(lambda: defaultdict(list))
    for ann in coco_dt:
        grouped[ann["image_id"]][ann["category_id"]].append(ann)
    for image_id, image in grouped.items():
        for category_id, anns in image.items():
            grouped[image_id][category_id] = sorted(anns, key=lambda x: x["score"], reverse=True)
    return grouped


def ungroup(grouped):
    coco_dt = []
    for image_id, image in grouped.items():
        for category_id, anns in image.items():
            for ann in anns:
                coco_dt.append(ann)
    return coco_dt


def main(args):
    with open(args.coco_gt_in, "r") as f:
        coco_gt = json.load(f)

    filename_to_image = {image["file_name"]: image for image in coco_gt["images"]}

    # отбор изображений
    print("Select images")
    image_ids, images_new = [], []
    for filename in tqdm(os.listdir(args.folder)):
        image = filename_to_image[filename]
        images_new.append(image)
        image_ids.append(image["id"])
    image_ids = set(image_ids)


    annotations_new, ann_cat_ids = [], []
    if args.csvs:  # выбор аннотаций из csv
        print("Load annotations from csvs")
        df = []
        for csv in args.csvs.split(","):
            df.append(pd.read_csv(csv))
        df = pd.concat(df, ignore_index=True)
        sign_class_to_id = {cat["name"]: cat["id"] for cat in coco_gt["categories"]}
        filename_to_id = {image["file_name"]: image["id"] for image in coco_gt["images"]}
        ann_to_id = {
            (ann["image_id"], ann["category_id"], tuple(ann["bbox"])): ann["id"]
            for ann in coco_gt["annotations"]
        }
        for rec in tqdm(df.to_dict("records")):
            image_id = filename_to_id[rec["filename"]]
            category_id = sign_class_to_id[rec["sign_class"]]
            x, y, w, h = rec["x_from"], rec["y_from"], rec["width"], rec["height"]
            ann = (image_id, category_id, (x, y, w, h))
            try:
                ann_id = ann_to_id[ann]
            except KeyError:
                print("Annotation for {} not found".format(rec["filename"]))
            annotation = dict(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                area=w * h,
                bbox=[x, y, w, h],
                iscrowd=0
            )
            annotations_new.append(annotation)
            ann_cat_ids.append(category_id)
    else:  # отбор аннотаций по id изображений
        print("Select annotations")
        for annotation in tqdm(coco_gt["annotations"]):
            if annotation["image_id"] in image_ids:
                annotations_new.append(annotation)
                ann_cat_ids.append(annotation["category_id"])
    ann_cat_ids = set(ann_cat_ids)

    categories_new = []
    if args.cats:  # загрузка категорий из внешнего фаила
        print("Load categories")
        with open(args.cats, "r") as f:
            categories_new = json.load(f)["categories"]
    elif args.rm_cats:  # удалений категорий, не представленных на отобранных изобраэениях
        print("Remove omitted categories")
        for category in coco_gt["categories"]:
            if category["id"] in ann_cat_ids:
                categories_new.append(category)

    print("Saving json")
    coco_gt = dict(images=images_new, annotations=annotations_new, categories=categories_new)
    with open(args.coco_gt_out, "w") as f:
        json.dump(coco_gt, f, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
