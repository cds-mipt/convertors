import argparse
import os
import json

from tqdm import tqdm

from utils import load_annot_as_df


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-root", type=str, required=True)
    parser.add_argument("--coco-gt-out", type=str, required=True)
    return parser


# можно переопределить так как нужно
# остальное вряд ли есть смысл менять
def get_category_name(cls, temporary, data):
    if cls in ["1.22", "1.23", "2.4", "2.5", "3.1", "3.2", "3.4", "3.24"]:
        return cls
    elif cls in ["3.25", "3.31"]:
        return "3.25+3.31"
    elif cls in ["5.19.1", "5.19.2", "5.19"]:
        return "5.19"
    else:
        return "other"


def main(args):
    names = []
    for annot_folder in tqdm(os.listdir(args.annot_root)):
        annot_path = os.path.join(args.annot_root, annot_folder)
        if not os.path.isdir(annot_path):
            continue
        for filename in os.listdir(annot_path):
            path = os.path.join(annot_path, filename)
            df = load_annot_as_df(path)
            for r in df.to_dict("record"):
                name = get_category_name(
                    cls=r["class"],
                    temporary=r["temporary"],
                    data=r["data"]
                )
                if name:
                    names.append(name)

    names = sorted(set(names))
    categories = [
        {"id": cat_id, "name": name, "supercategory": "traffic sign"}
        for cat_id, name in enumerate(names, start=1)
    ]

    coco_gt = dict(categories=categories)
    with open(args.coco_gt_out, "w") as f:
        json.dump(coco_gt, f)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
