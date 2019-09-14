import argparse
import os
import json
import cv2
import numpy as np
import pycocotools.mask as mask_utils

from PIL import Image
from skimage.measure import label


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--coco-gt-out", type=str, required=True)
    return parser


def create_images_field(image_folder):
    images = []
    for id_, filename in enumerate(os.listdir(image_folder)):
        path = os.path.join(image_folder, filename)

        width, height = Image.open(path).size
        image = dict(id=id_, width=width, height=height, file_name=filename)

        images.append(image)
    return images


def load_bboxes(path):
    bboxes = []
    with open(path, "r") as f:
        for line in f:
            name, x1, y1, x2, y2 = line.strip().split(" ")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bboxes.append((name, [x1, y1, x2 - x1, y2 - y1]))
    return bboxes


def create_annotations_field(images, categories, labels_folder, mask_folder):
    name_to_cat_id = {cat["name"]: cat["id"] for cat in categories}
    annotations = []
    ann_id = 1  # start from 1 !
    for image in images:
        name = image["file_name"].split(".")[0]

        bboxes = load_bboxes(os.path.join(labels_folder, name + ".txt"))
        mask = cv2.imread(os.path.join(mask_folder, name + ".bmp"), cv2.IMREAD_GRAYSCALE)
        for cat_name, bbox in bboxes:
            x, y, w, h = bbox
            if w * h > 0:
                segmentation = extract_instance_mask(mask, bbox)
                if segmentation:
                    segmentation["counts"] = segmentation["counts"].decode()
                    annotation = dict(
                        id=ann_id,
                        image_id=image["id"],
                        category_id=name_to_cat_id[cat_name],
                        segmentation=segmentation,
                        area=w * h,
                        bbox=bbox,
                        iscrowd=0
                    )
                    ann_id += 1
                    annotations.append(annotation)

    return annotations


def extract_instance_mask(mask, bbox):
    x, y, w, h = bbox
    instance_mask = (mask[y:y + h, x:x + w] > 128).astype(np.uint8)
    labeled, n = label(instance_mask, return_num=True)

    if n > 0:
        areas = []
        for l in range(1, n + 1):
            instance = (labeled == l).astype(np.uint8)
            proj_x_size = (instance.sum(axis=0) > 0).sum()
            proj_y_size = (instance.sum(axis=1) > 0).sum()
            areas.append(proj_x_size * proj_y_size)

        instance_label = np.argmax(areas) + 1
        instance_mask = (labeled == instance_label).astype(np.uint8)
        instance_mask_global = np.zeros_like(mask)
        instance_mask_global[y:y + h, x:x + w] = instance_mask
        return mask_utils.encode(np.asfortranarray(instance_mask_global))


def main(args):
    color_folder = os.path.join(args.folder, "color")
    labels_folder = os.path.join(args.folder, "labels")
    mask_folder = os.path.join(args.folder, "mask")

    images = create_images_field(color_folder)
    categories = [{"name": "car", "id": 1, "supercategory": "vehicle"}]
    annotations = create_annotations_field(images, categories, labels_folder, mask_folder)

    coco_gt = dict(images=images, categories=categories, annotations=annotations)
    with open(args.coco_gt_out, "w") as f:
        json.dump(coco_gt, f, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
