# convertors
## convert viva to coco
Конвертирует файлы из Viva формата в coco .json. Конвертирует не отдельный xml, а весь датасет сразу(dayTrain, nightTrain).
Если у вас не стандартный Viva датасет, то он должен иметь такую структуру:

root_folder -> example_folder1 -> *BOX.csv.\
root_folder -> example_folder1 -> example_folder2 -> *.png,
где root_folder - это папка, которую вы указываете в аргументах  командной строки. Если такое расположение для вас неудобно, можете поменять их в методе get_immages_and_annot.

                        
