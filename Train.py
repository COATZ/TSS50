import os
import sys
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import fcn2
import utils

# ctrees = [11, 236, 9]
# cground = [187, 70, 156]
# csky = [29, 26, 199]
# cblack = [0, 0, 0]
# seg_color = [cblack, ctrees, cground, csky]

ctrees = [11, 236, 9]
cground = [187, 70, 156]
csky = [29, 26, 199]
# cblack = [0, 0, 0]
seg_color = [ctrees, cground, csky]

Learning_Rate = 1e-5
width = height = 100  # image width and height
batchSize = 10
nb_iter = 1000
nodilation = True
opt = "nodil"
version = 1

TrainFolder = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_3/CUBE/"
ListImages = os.listdir(os.path.join(TrainFolder, "RGB"))  # Create list of images

# ----------------------------------------------Transform image-------------------------------------------------------------------
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),
                          tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn = tf.Compose([tf.ToPILImage(), tf.Resize((height, width), tf.InterpolationMode.NEAREST), tf.ToTensor()])

# ---------------------Metrics logger ---------------------------------------------------------
acc_meter = utils.AverageMeter()
intersection_meter = utils.AverageMeter()
union_meter = utils.AverageMeter()

# ---------------------Read image ---------------------------------------------------------


def ReadRandomImage():  # First lets load random image and  the corresponding annotation
    idx = np.random.randint(0, len(ListImages))  # Select random image
    Img = cv2.imread(os.path.join(TrainFolder, "RGB", ListImages[idx]))[:, :, 0:3]
    Seg = cv2.imread(os.path.join(TrainFolder, "SEG", ListImages[idx].replace(
        "_rgb.png", "_seg.png")))[:, :, ::-1]
    # Trees = cv2.imread(os.path.join(TrainFolder, "TREE", ListImages[idx].replace(
    #     "_rgb.png", "_seg.png")), cv2.IMREAD_COLOR)
    # Sky = cv2.imread(os.path.join(TrainFolder, "SKY", ListImages[idx].replace(
    #     "_rgb.png", "_seg.png")), cv2.IMREAD_COLOR)
    # Ground = cv2.imread(os.path.join(TrainFolder, "GROUND", ListImages[idx].replace(
    #     "_rgb.png", "_seg.png")), cv2.IMREAD_COLOR)

    AnnMap = np.zeros(Seg.shape[0:2], np.float32)
    for idx in range(len(seg_color)):
        AnnMap[np.where((Seg == seg_color[idx]).all(axis=2))] = idx
    # utils.get_unique2(AnnMap)
    Img = transformImg(Img)
    AnnMap = transformAnn(AnnMap)
    return Img, AnnMap


# --------------Load batch of images-----------------------------------------------------
def LoadBatch():  # Load batch of images
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i], ann[i] = ReadRandomImage()
    return images, ann


# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
# Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
# Net = torchvision.models.segmentation.fcn_resnet50(pretrained=True)  # Load net
Net = fcn2.fcn_resnet50(pretrained=True, nodilation=nodilation, )  # Load net
Net.classifier[4] = torch.nn.Conv2d(512, len(seg_color), kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes


net = "training_"+str(opt)+"_b"+str(batchSize)+"_i"+str(nb_iter)+"_v"+str(int(version))
print(Net)
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer

# ----------------Train--------------------------------------------------------------------------
acc_ref = 0
for itr in range(nb_iter):  # Training loop
    images, ann = LoadBatch()  # Load taining batch
    images = torch.autograd.Variable(images, requires_grad=False).to(device)  # Load image
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)  # Load annotation
    Pred = Net(images)['out']  # make prediction
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    if itr % 10 == 0:
        # acc_meter.initialize(0)
        # intersection_meter.initialize(0)
        # union_meter.initialize(0)

        acc_list = np.array([])
        intersection_list = np.array([])
        union_list = np.array([])
        iou_list = np.array([])

        for idx in range(batchSize):
            seg_pred = torch.argmax(Pred[idx], 0).cpu().detach().numpy()[..., np.newaxis]  # Get  prediction classes
            ann_test = ann[idx].cpu().detach().numpy()[..., np.newaxis]
            acc, pix = utils.accuracy(seg_pred, ann_test)
            intersection, union = utils.intersectionAndUnion(seg_pred, ann_test, len(seg_color))
            # acc_meter.update(acc, pix)
            # intersection_meter.update(intersection)
            # union_meter.update(union)

            iou = intersection / (union + 1e-10)
            acc_list = np.append(acc_list, acc)
            intersection_list = np.append(intersection_list, intersection)
            union_list = np.append(union_list, intersection)
            iou_list = np.append(iou_list, iou)
            # print(iou_list)

        # miou = intersection_meter.sum / (union_meter.sum + 1e-10)
        print("Iter {} Loss {} Acc {} MIoU {}".format(itr, Loss.data.cpu().numpy(), np.mean(acc_list), np.mean(iou_list)))

        if (np.mean(acc_list) >= acc_ref and itr > int(nb_iter/3)):
            print("BEST NEW at iter "+str(itr)+": saving model")
            acc_ref = np.mean(acc_list)
            torch.save(Net.state_dict(), os.path.join("./ckpt/", str(net) + "_best.torch"))

print("Saving Model" + str(net) + ".torch")
torch.save(Net.state_dict(), os.path.join("./ckpt/", str(net) + ".torch"))
