import os
import sys
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import fcn2

ctrees = [11, 236, 9]
cground = [187, 70, 156]
csky = [29, 26, 199]
cblack = [0, 0, 0]

Learning_Rate = 1e-5
width = height = 100  # image width and height
batchSize = 3

TrainFolder = "/media/cartizzu/DATA/DATASETS/GRID_425_DATA_1_bil_test_1/CUBE/"
ListImages = os.listdir(os.path.join(TrainFolder, "RGB"))  # Create list of images

#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),
                          tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn = tf.Compose([tf.ToPILImage(), tf.Resize((height, width), tf.InterpolationMode.NEAREST), tf.ToTensor()])

#---------------------Read image ---------------------------------------------------------


def ReadRandomImage():  # First lets load random image and  the corresponding annotation
    idx = np.random.randint(0, len(ListImages))  # Select random image
    Img = cv2.imread(os.path.join(TrainFolder, "RGB", ListImages[idx]))[:, :, 0:3]
    Trees = cv2.imread(os.path.join(TrainFolder, "TREE", ListImages[idx].replace(
        "_rgb.png", "_seg.png")), cv2.IMREAD_COLOR)
    # arr = np.unique(Trees.reshape(-1, 3), axis=0)
    # print(arr)
    # print(Trees)
    # Vessel = cv2.imread(os.path.join(TrainFolder, "Semantic/1_Vessel", ListImages[idx].replace("jpg", "png")), 0)
    AnnMap = np.zeros(Img.shape[0:2], np.float32)
    if Trees is not None:
        AnnMap[Trees[..., 0] == ctrees[0]] = 1
    # if Trees is not None:
    #     AnnMap[Trees == 1] = 2
    Img = transformImg(Img)
    AnnMap = transformAnn(AnnMap)
    # print(AnnMap)
    # arr = np.unique(AnnMap)
    # print(arr)
    # sys.exit()
    return Img, AnnMap

#--------------Load batch of images-----------------------------------------------------


def LoadBatch():  # Load batch of images
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i], ann[i] = ReadRandomImage()
    return images, ann


#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
# Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
Net = torchvision.models.segmentation.fcn_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
# Net = fcn2.fcn_resnet18()  # Load net
# Net.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes

net = "fcn_resnet50"
print(Net)
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer

#----------------Train--------------------------------------------------------------------------
for itr in range(10000):  # Training loop
    images, ann = LoadBatch()  # Load taining batch
    images = torch.autograd.Variable(images, requires_grad=False).to(device)  # Load image
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)  # Load annotation
    Pred = Net(images)['out']  # make prediction
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get  prediction classes
    if itr % 100 == 0:
        print(itr, ") Loss=", Loss.data.cpu().numpy())
    if itr % 1000 == 0:  # Save model weight once every 60k steps permenant file
        print("Saving Model" + str(itr) + ".torch")
        torch.save(Net.state_dict(),   str(itr) + ".torch")

print("Saving Model" + str(net) + ".torch")
torch.save(Net.state_dict(), str(net) + ".torch")
