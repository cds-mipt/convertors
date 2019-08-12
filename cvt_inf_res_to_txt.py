from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import glob
import time
import numpy as np
from tqdm import tqdm

test_im = "/home/serg_t/Documents/datasets/dayTrain/dayClip1/frames/dayClip1--00000.png"
checkpoint_file = '/home/serg_t/Documents/mmdetection/cascade_rcnn_2/epoch_12.pth'
config_file = '/home/serg_t/Documents/mmdetection/configs/cascade_rcnn_r50_fpn_1x_viva.py'

def cvt_inf_res_to_txt(img_name, output_dest=None, model=None, bb_cnt=0):
    result = inference_detector(model, img_name)
    #print(result)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    #print(bboxes)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    labels_str = []
    if isinstance(model.CLASSES, list):
        for l in labels:
            labels_str.append(model.CLASSES[l])
    else:
        for l in labels:
            labels_str.append(model.CLASSES)
    #print(labels_str)
    if output_dest is not None:
        out_file = output_dest + '/'  + img_name.split('/')[-1].split('.')[0] + '.txt'
        with open(out_file, "w") as file:
            info = []
            for i, label in enumerate(labels_str):
                if label == "traffic_light":
                    info.append(label + ' ' + str(bboxes[i][4]) + ' ' + ' '.join(map(str, bboxes[i][0:4])) + '\n')

            file.writelines(info)

if __name__ == '__main__':
    #dayTrain
    bb_cnt = 0
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    dest_folder = "/home/serg_t/Documents/datasets/dayVal/converted_inf_res/"
    imgs_list  = glob.glob("/home/serg_t/Documents/datasets/dayVal/*/*/*.png", recursive=True)
    #print(imgs_list[0])
    #cvt_inf_res_to_txt(test_im, dest_folder)
    for im in tqdm(imgs_list):
        cvt_inf_res_to_txt(im, dest_folder, model=model)