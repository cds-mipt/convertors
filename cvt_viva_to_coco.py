import glob
import pandas as pd
import json
import numpy as np


PRE_DEFINED_CATEGORIES = {'traffic_light':1}#Here we put our categ

#returns list of dict
def get_images_and_annot(dataset_folder, img_size=(1280, 960), img_format="png"):
    img_list = glob.glob(dataset_folder + "/*/*/*.png", recursive=True)
    #get list of dict with image info
    image_list_dict = []
    img_list = sorted(img_list)
    for i, im in enumerate(img_list):
        img_list[i] = im.split("/")[-1]
        im_id = i
        width = img_size[0]
        height = img_size[1]
        image_list_dict.append({'file_name':img_list[i], 'id':im_id, "width":width, "height":height})
    #get list of dicts with annot info
    csv_list = sorted(glob.glob(dataset_folder + '/*/*BOX.csv'))
    annot_list = []
    for csv in csv_list:
        tb = pd.read_csv(csv, delimiter=";")
        ann_cnt = 0
        for i, im_name in enumerate(tb.loc[:, 'Filename']):
            im_name = im_name.split("/")[-1]
            image_id = img_list.index(im_name)
            x = np.int(tb.iloc[i, 2])
            y = np.int(tb.iloc[i, 3])
            box_w = np.int(tb.iloc[i, 4] - tb.iloc[i, 2])
            box_h = np.int(tb.iloc[i, 5] - tb.iloc[i, 3])
            area = box_h * box_w
            #iscrowd always 0 and categ_id always 1(traffic_light)
            annot_list.append({"image_id":image_id, "area":area, "bbox":[x, y, box_w, box_h], "id": ann_cnt, 'category_id': 1, 'iscrowd':0})
            ann_cnt += 1

    return image_list_dict, annot_list

def get_categories(dict_of_categ=PRE_DEFINED_CATEGORIES):
    json_cat_list = []
    for cate, cid in dict_of_categ.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_cat_list.append(cat)
    return json_cat_list

def get_annot_into_json(dataset_path, output_file):
    images, annotations = get_images_and_annot(dataset_path)
    categories = get_categories()
    json_dict = {'images':images, 'annotations':annotations, 'categories':categories}
    json_fp = open(output_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == '__main__':
    #im , annot = get_images_and_annot("/home/serg_t/Documents/datasets/dayTrain")
    #print(im[0:10], "\n", annot[0:10])
    #print(len(annot))
    get_annot_into_json("/home/serg_t/Documents/datasets/dayVal", "/home/serg_t/Documents/mmdetection/data/viva/annotations/val_annotations.json")
