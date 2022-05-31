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

ctrees = [11, 236, 9]
cground = [187, 70, 156]
csky = [29, 26, 199]
cblack = [0, 0, 0]
seg_color = [cblack, ctrees, cground, csky]

imagePath = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_1/CUBE/TEST/"
# img_loc = "1_0_back_custom_rgb.png"
# img_loc = "42_0_back_custom_rgb.png"
img_loc = "1_0_rgb.png"
height = width = 100

#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(), tf.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # tf.Resize((300,600)),tf.RandomRotation(145)])#

#---------------------Metrics logger ---------------------------------------------------------
acc_meter = utils.AverageMeter()
intersection_meter = utils.AverageMeter()
union_meter = utils.AverageMeter()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
# # Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
# # Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
# Net = fcn2.fcn_resnet50(pretrained=True)  # Load net
Net = fcn2_sphe.fcn_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
# # Net = fcn2.fcn_resnet18()  # Load net
# # Net.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes

net = "fcn_resnet50_nodilation2"
modelPath = os.path.join("./ckpt", str(net) + '.torch')
Net.load_state_dict(torch.load(modelPath))  # Load trained model
Net = Net.to(device)  # Set net to GPU or CPU
Net.eval()  # Set to evaluation mode
print(Net)

acc_meter.initialize(0)
intersection_meter.initialize(0)
union_meter.initialize(0)

img_filename = os.path.join(imagePath, img_loc)
Img = cv2.imread(img_filename)  # load test image
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

gt_img = cv2.imread(os.path.join(imagePath, img_loc).replace("_rgb.png", "_seg.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
gt_label = np.zeros((height, width))
for idx in range(len(seg_color)):
    gt_label[np.where((gt_img == seg_color[idx]).all(axis=2))] = idx
acc, pix = utils.accuracy(seg_pred, gt_label)
intersection, union = utils.intersectionAndUnion(seg_pred, gt_label, len(seg_color))
acc_meter.update(acc, pix)
intersection_meter.update(intersection)
union_meter.update(union)

iou = intersection_meter.sum / (union_meter.sum + 1e-10)

print("Acc {} MIoU {} IoU {}".format(acc_meter.average(), iou.mean(), iou))

plt.imshow(seg_rgb)  # display image
plt.show()
