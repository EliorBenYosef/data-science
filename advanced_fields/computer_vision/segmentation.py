"""
Deep Learning Exercise
The following code implements a segmentation algorithm based on a fully connected network with resnet101 base.
We aim to implement a person detection stream that extracts a json-file with all detections for a given video.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import imageio
import matplotlib.pyplot as plt
# from urllib.request import urlopen
# from IPython.display import clear_output
# from time import sleep

############################

# Load pretrained segmentation model.

model = models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21)
model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


############################

# Part 1
# Implement a function to extracts the sementaic segmentation for the class 'person' for a given image.
# You may import a local image to colab by choosing files on the left taskbar. Obtain the file path by right-click.
# Verify you transform the image using the TF.

def decode_segmap(seg):
    r = np.zeros_like(seg).astype(np.uint8)
    g = np.zeros_like(seg).astype(np.uint8)
    b = np.zeros_like(seg).astype(np.uint8)

    mask = seg == 1
    # r[mask] = 0
    # g[mask] = 0
    b[mask] = 128

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment_person(img):
    """
    Gets a single image with 3 channels (RGB)
    Returns semantic segmentation for the class 'person'
    """
    input = tf(img).unsqueeze(0)
    output = model(input)['out'].squeeze()

    # to get a single muiti-class image:
    clss_img = torch.argmax(output, dim=0).detach().cpu().numpy()
    # print (np.unique(clss_img))  # how many classes are in the img

    # Classes:
    # 0=background,
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person,
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor.

    # to get a single binary (single class) image (with 0\1 if a person):
    seg = np.zeros_like(clss_img).astype(np.uint8)
    mask = clss_img == 15  # person_clss
    seg[mask] = 1

    return seg


img_pil = Image.open('../../datasets/per_field/cv/people_in_office.jpg')
img_pil = img_pil.convert('RGB')  # if the img has an additional alpha channel

seg = segment_person(img_pil)

rgb = decode_segmap(seg)
plt.imshow(rgb)
plt.show()


############################

# Part 2
# Implement a function that loads a video and returns a json file with position (bounding-box corners \ contours)
# of each person per timeframe. The video is loaded as gif and may be processed frame-wise.

def obtain_tags(video):
    """
    return json with segmentation results
    """

    jsonDict = {}

    try:
        while 1:
            video.seek(video.tell() + 1)
            timestamp = video.tell()
            img = video.convert('RGB')
            seg = segment_person(img)
            image, contours, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            jsonDict[timestamp] = contours

    except EOFError:
        pass

    return jsonDict


# url = 'https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/preview/face-demographics-walking.gif'
# video = Image.open(urlopen(url))
video = Image.open('../../datasets/per_field/cv/walking.gif')

json = obtain_tags(video)


############################

# Part 3
# Visualize the previously extracted json results on the original video (e.g. by bounding boxes or contures)

def visualize(jsonFile, video):
    """
    overlay video with json results
    """

    try:
        while True:
            video.seek(video.tell() + 1)

            timeframe = video.tell()
            frame = video.convert('RGB')

            img_cont = np.array(frame)
            cv2.drawContours(img_cont, jsonFile[timeframe], -1, (0, 255, 0), 3)
            images.append(img_cont)

            # clear_output()
            # cv2_imshow(img_cont)
            # sleep(0.1)

    except EOFError:
        pass


# url = 'https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/preview/face-demographics-walking.gif'
# video = Image.open(urlopen(url))
video = Image.open('../../datasets/per_field/cv/walking.gif')

images = []
visualize(json, video)
imageio.mimsave('/content/walking_seg.gif', images)
