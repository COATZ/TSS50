import os
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import fcn2
import numpy as np
import utils

ctrees = [11, 236, 9]
cground = [187, 70, 156]
csky = [29, 26, 199]
cblack = [0, 0, 0]
seg_color = [ctrees, cground, csky]

imagePath = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_1/CUBE/TEST/"
img_loc = "1_0_back_custom_rgb.png"

height = width = 100
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(), tf.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # tf.Resize((300,600)),tf.RandomRotation(145)])#


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
# Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
# Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
Net = torchvision.models.segmentation.fcn_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
# Net = fcn2.fcn_resnet18()  # Load net
# Net.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes

net = "fcn_resnet50"
Net = Net.to(device)  # Set net to GPU or CPU
modelPath = str(net) + '.torch'
Net.load_state_dict(torch.load(modelPath))  # Load trained model
Net.eval()  # Set to evaluation mode
# print(Net)

img_filename = os.path.join(imagePath, img_loc)
Img = cv2.imread(img_filename)  # load test image
# plt.imshow(Img[:, :, ::-1])  # Show image
# plt.show()

Img = transformImg(Img)  # Transform to pytorch
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    Prd = Net(Img)['out']  # Run net
Prd = tf.Resize((height, width))(Prd[0])  # Resize to origninal size
seg = torch.argmax(Prd, 0).cpu().detach().numpy()[..., np.newaxis]  # Get  prediction classes

gt_img = cv2.imread(os.path.join(imagePath, img_loc).replace("_rgb.png", "_seg.png"), cv2.IMREAD_COLOR)
gt_label = np.zeros((height, width))
gt_label[gt_img[..., 0] == ctrees[0]] = 1
acc = utils.accuracy(seg, gt_label)
assert acc[-1] == int(height * width), "Not all pixels used in acc computation"
print("Accuracy {}".format(acc[0]))

seg_rgb = utils.colorEncode(seg, seg_color)
miou = utils.intersectionAndUnion(seg_rgb, gt_img, 3)
print("MIoU {}".format(acc[0]))

plt.imshow(seg_rgb)  # display image
plt.show()
