import os
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import fcn2
import fcn2_sphe
import numpy as np
import utils

from operator import le, ge, eq

ctrees = [11, 236, 9]
cground = [187, 70, 156]
csky = [29, 26, 199]
# cblack = [0, 0, 0]
seg_color = [ctrees, cground, csky]

# imagePath = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_1/CUBE/TEST/"
test_path = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_3/EQUI/TEST_100"
# save_dir = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_3/EQUI/OUTPUT/"
height = width = 100
batchSize = 10
nb_iter = 1000
nodilation = True
opt = "nodil"
version = 1


#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(), tf.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # tf.Resize((300,600)),tf.RandomRotation(145)])#

#---------------------Metrics logger ---------------------------------------------------------
acc_meter = utils.AverageMeter()
intersection_meter = utils.AverageMeter()
union_meter = utils.AverageMeter()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Check if there is GPU if not set trainning to CPU (very slow)


def test(Net, net_opt):

    acc_list = np.array([])
    intersection_list = np.zeros(len(seg_color))
    union_list = np.zeros(len(seg_color))
    iou_list = np.zeros(len(seg_color))

    Net.classifier[4] = torch.nn.Conv2d(512, len(seg_color), kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
    net = "training_"+str(opt)+"_b"+str(batchSize)+"_i"+str(nb_iter)+"_v"+str(int(version))
    modelPath = os.path.join("./ckpt", str(net) + '.torch')
    # modelPath = os.path.join("./ckpt", str(net) + '_best.torch')
    Net.load_state_dict(torch.load(modelPath))  # Load trained model
    Net = Net.to(device)  # Set net to GPU or CPU
    Net.eval()  # Set to evaluation mode
    print(Net)

    save_dir = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_3/EQUI/OUTPUT/"
    save_dir = os.path.join(save_dir, net)
    os.makedirs(save_dir, exist_ok=True)

    acc_meter.initialize(0)
    intersection_meter.initialize(0)
    union_meter.initialize(0)

    for idy, file in enumerate([f for f in sorted(os.listdir(test_path)) if (os.path.isfile(os.path.join(test_path, f)) and f.endswith('_rgb.png'))]):
        # print("Processing IMG ", file)
        img_filename = os.path.join(test_path, file)
        # Img = cv2.imread(img_filename)  # load test image
        # Img_rgb = np.array(Img)

        Img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
        RGB_img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img_rgb = np.array(RGB_img)
        # plt.imshow(Img[:, :, ::-1])  # Show image
        # plt.show()

        Img = transformImg(Img)  # Transform to pytorch
        Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
        with torch.no_grad():
            Prd = Net(Img)['out']  # Run net
        Prd = tf.Resize((height, width))(Prd[0])  # Resize to origninal size
        seg_pred = torch.argmax(Prd, 0).cpu().detach().numpy()[..., np.newaxis]  # Get  prediction classes
        # utils.get_unique2(seg_pred)
        seg_rgb = utils.colorEncode(seg_pred, seg_color)
        # plt.imshow(seg_rgb)  # Show image
        # plt.show()

        gt_img = cv2.imread(os.path.join(test_path, file).replace("_rgb.png", "_seg.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        gt_label = np.zeros((height, width))
        for idx in range(len(seg_color)):
            gt_label[np.where((gt_img == seg_color[idx]).all(axis=2))] = idx
        acc, pix = utils.accuracy(seg_pred, gt_label)
        intersection, union = utils.intersectionAndUnion(seg_pred, gt_label, len(seg_color))
        iou = intersection / (union + 1e-10)

        acc_list = np.append(acc_list, acc)
        intersection_list = np.vstack((intersection_list, intersection))
        union_list = np.vstack((union_list, union))
        iou_list = np.vstack((iou_list, iou))
        if np.max(iou) > 1:
            print("Alert IOU > 1: {} for IMAGE {}".format(iou, file))

        numpy_vertical = np.vstack((Img_rgb, seg_rgb))
        cv2.imwrite(os.path.join(save_dir, file[:-4] + "_" + str(net_opt) + ".png"), cv2.cvtColor(numpy_vertical, cv2.COLOR_RGB2BGR))

        # print(intersection, union)
        # acc_meter.update(acc, pix)
        # intersection_meter.update(intersection)
        # union_meter.update(union)

        # if file == "1_0_rgb.png":
        #     numpy_vertical = np.vstack((Img_rgb, seg_rgb))

    intersection_list = np.delete(intersection_list, 0, 0)
    union_list = np.delete(union_list, 0, 0)
    iou_list = np.delete(iou_list, 0, 0)
    if len(seg_color) == 4:
        intersection_list = np.delete(intersection_list, 0, -1)
        union_list = np.delete(union_list, 0, -1)
        iou_list = np.delete(iou_list, 0, -1)

    # iou = intersection_meter.sum / (union_meter.sum + 1e-10)

    # print(iou_list)
    # print("Acc {} MIoU {} IoU {}".format(acc_meter.average(), iou.mean(), iou))
    # print("Acc {} MIoU {} IoU {}".format(np.mean(acc_list), np.mean(iou_list), np.mean(iou_list, axis=0)))

    # plt.imshow(numpy_vertical)  # display image
    # plt.show()

    return acc_list, iou_list


def main():
    nodilation = True
    iFCNHead_sphe = True
    iFL_sphe = False
    LAYER_CCOUNT = 0
    iSYMBOL = le

    # # Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
    # # Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
    Net_ref = fcn2.fcn_resnet50(pretrained=True, nodilation=nodilation,)  # Load net
    Net = fcn2_sphe.fcn_resnet50(pretrained=True,
                                 nodilation=nodilation,
                                 iFCNHead_sphe=iFCNHead_sphe,
                                 iFL_sphe=iFL_sphe,
                                 iSYMBOL=iSYMBOL,
                                 LAYER_CCOUNT=LAYER_CCOUNT)  # Load net

    acc_list_0, iou_list_0 = test(Net_ref, "ref")
    acc_list_1, iou_list_1 = test(Net, "full")

    print("Acc {} MIoU {} IoU {}".format(np.mean(acc_list_0), np.mean(iou_list_0), np.mean(iou_list_0, axis=0)))
    print("Acc {} MIoU {} IoU {}".format(np.mean(acc_list_1), np.mean(iou_list_1), np.mean(iou_list_1, axis=0)))

if __name__ == '__main__':
    main()
