import os
# import sys
import cv2
import numpy as np
import shutil

dataset_path = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil/drone_training_ppo_2022_03_11_17_48_30/CUBE/"
new_dataset_path = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_3/"

os.makedirs(new_dataset_path, exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "SEG"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "RGB"), exist_ok=True)

ctrees = [11, 236, 9]
cground = [187, 70, 156]
csky = [29, 26, 199]
cblack = [0, 0, 0]

list_color = np.array([[]])
ccounter = 0
nb_max_img = 5000
for file in [f for f in os.listdir(dataset_path) if (os.path.isfile(os.path.join(dataset_path, f)) and (f.endswith('_front_custom_seg.png') or f.endswith('_back_custom_seg.png')))]:
    ccounter += 1
    if int(file.split('_')[0]) <= nb_max_img:
    # if int(file.split('_')[0]) == 42:
        img = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_COLOR)[..., ::-1]
        for idx, lbl in enumerate(["TREE", "SKY", "GROUND"]):
            os.makedirs(os.path.join(new_dataset_path, str(lbl)), exist_ok=True)
            img_copy = np.zeros((img.shape))
            for h in range(img.shape[0]):
                for w in range(img.shape[1]):
                    chan = img[h, w]
                    if lbl == "TREE":
                        pix_test = ctrees
                    elif lbl == "SKY":
                        pix_test = csky
                    elif lbl == "GROUND":
                        pix_test = cground
                    if (chan[0] == pix_test[0] and chan[1] == pix_test[1] and chan[2] == pix_test[2]):
                        img_copy[h, w] = pix_test[::-1]

            cv2.imwrite(os.path.join(new_dataset_path, str(lbl), file), img_copy)

        seg_filename = file
        shutil.copyfile(os.path.join(dataset_path, seg_filename), os.path.join(new_dataset_path, "SEG", seg_filename))

        rgb_filename = file.split('_seg.png')[0]+"_rgb.png"
        shutil.copyfile(os.path.join(dataset_path, rgb_filename), os.path.join(new_dataset_path, "RGB", rgb_filename))


print(len(os.listdir(os.path.join(new_dataset_path, "RGB"))))
