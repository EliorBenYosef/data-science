"""
Deep Learning Image Segmentation
Implementing a segmentation algorithm based on a fully-connected neural network with resnet101 base.
Implementing a person detection stream that extracts a json-file with all detections for a given video.
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

# Load pretrained segmentation model:
# 21 Classes:
#   0=background,
#   1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
#   6=bus, 7=car, 8=cat, 9=chair, 10=cow,
#   11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person,
#   16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor.
model = models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21)
model.eval()
chosen_clss = 15  # person_clss

# Create an image transformer:
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def segment_person(img):
    """
    :param img: a single image with 3 channels (RGB)
    :return: a semantic segmentation for the class 'person' for the given image
    """
    input = tf(img).unsqueeze(0)  # transform the image using the TF
    output = model(input)['out'].squeeze()

    # to get a single muiti-class image:
    clss_img = torch.argmax(output, dim=0).detach().cpu().numpy()
    # print (np.unique(clss_img))  # how many classes are in the img

    # to get a single binary (single class) image (with 0\1 if matches the class):
    seg = np.zeros_like(clss_img).astype(np.uint8)
    mask = clss_img == chosen_clss
    seg[mask] = 1

    return seg


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


def obtain_tags(video):
    """
    :param video:
    :return: a json file with segmentation results:
        position (bounding-box corners / contours) of each person per timeframe.
    """
    seg_json = {}

    try:
        while 1:  # The video is processed frame-wise:
            video.seek(video.tell() + 1)
            timestamp = video.tell()
            img = video.convert('RGB')
            seg = segment_person(img)
            contours, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            seg_json[timestamp] = contours

    except EOFError:
        pass

    return seg_json


def visualize(video, seg_json):
    """
    Overlay video with json results
    Visualize the extracted json results on the original video (e.g. by bounding boxes or contours)
    """
    images = []

    try:
        while True:
            video.seek(video.tell() + 1)

            timeframe = video.tell()
            frame = video.convert('RGB')

            img_cont = np.array(frame)
            cv2.drawContours(img_cont, seg_json[timeframe], -1, (0, 255, 0), 3)
            images.append(img_cont)

            # clear_output()
            # cv2_imshow(img_cont)
            # sleep(0.1)

    except EOFError:
        pass

    return images


if __name__ == '__main__':
    img_pil = Image.open('../../../../datasets/per_field/cv/color_office.jpg')
    img_pil = img_pil.convert('RGB')  # if the img has an additional alpha channel
    seg = segment_person(img_pil)
    rgb = decode_segmap(seg)
    plt.imshow(rgb)
    plt.savefig('results/segmented_person.png')
    plt.show()

    # url = 'https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/preview/face-demographics-walking.gif'
    # video = Image.open(urlopen(url))
    video = Image.open('../../../../datasets/per_field/cv/walking.gif')  # loads a video
    seg_json = obtain_tags(video)
    images = visualize(video, seg_json)
    imageio.mimsave('results/walking_seg.gif', images)
