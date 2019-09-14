import pandas as pd
import glob
import argparse

def build_parser():
    parser = argparse.ArgumentParser("Convert viva to txt(Object Detection Metrics)")
    parser.add_argument("--in_path",
                        type=str,
                        help="Path to viva .xml file")
    parser.add_argument("--out_dir",
                        type=str,
                        help="Output dir for .txt files")
    return parser

def cvt_csv_to_txt(ann_file, out_path):
    t = pd.read_csv(ann_file, delimiter=";")
    #print(t)
    #print(t.iloc[:, 3])
    s = set()
    for i in range(t.shape[0]):
        file_name = t.iloc[i, 0].split(r'/')[-1].split('.')[0] + ".txt"
        info = ' '.join(['traffic_light', str(t.iloc[i, 2]), str(t.iloc[i, 3]), str(t.iloc[i, 4]), str(t.iloc[i, 5])])

        if file_name in s:
            with open(out_path + '/' + file_name, "a") as file:
                file.writelines(info + '\n')
        else:
            with open(out_path + '/' + file_name, "w") as file:
                file.writelines(info + '\n')
            s.add(file_name)




        #print(file_name)


if __name__ == '__main__':
    #convert dayTrain
    parser = build_parser()
    args = parser.parse_args()
    dest_folder = args.out_dir#"/home/serg_t/Documents/datasets/test_train/txt/"
    #list_of_annot = glob.glob("/home/serg_t/Documents/datasets/test_train/*/*BOX.csv", recursive=True)
    #list_of_annot = sorted(list_of_annot)
    #for ann in list_of_annot:
    #    cvt_csv_to_txt(ann, dest_folder)
    ann = args.in_path
    cvt_csv_to_txt(ann, dest_folder)