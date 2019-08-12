from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import glob
import time
import numpy as np
from tqdm import tqdm
import argparse

#test_im = #"/home/serg_t/Documents/datasets/dayTrain/dayClip1/frames/dayClip1--00000.png"

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

def build_parser():
    parser = argparse.ArgumentParser('Get inferense results in txt format')
    parser.add_argument("--img_dir",
                        type=str,
                        help='Directory with images')
    parser.add_argument("--dest_dir",
                        type=str,
                        help='Dir for output txt files')
    parser.add_argument('--im_format',
                        type=str,
                        help='images format')
    parser.add_argument('--checkpoint_file',
                        type=str,
                        help='.pth file with weights')
    parser.add_argument('--config_file',
                        type=str,
                        help='.py config file for model')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    checkpoint_file =  args.checkpoint_file# '/home/serg_t/Documents/mmdetection/cascade_rcnn_2/epoch_12.pth'
    config_file = args.config_file# '/home/serg_t/Documents/mmdetection/configs/cascade_rcnn_r50_fpn_1x_viva.py'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    dest_folder = args.dest_dir#"/home/serg_t/Documents/datasets/dayVal/converted_inf_res/"
    img_folder = args.img_dir
    im_format = args.im_format
    imgs_list  = glob.glob(img_folder + "*." + im_format, recursive=True)
    for im in tqdm(imgs_list):
        cvt_inf_res_to_txt(im, dest_folder, model=model)