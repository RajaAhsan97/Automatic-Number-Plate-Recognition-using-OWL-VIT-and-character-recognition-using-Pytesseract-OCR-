"""
    license plate recognition using Open world localization - vision Transformer (OWL-ViT) by hugging face, a pre-trained model for 
    detecting License plate in the given image. Furthermore, text in the detected region of llcense plate is recognized using a 
    Tesseract Optical Character Recognition (OCR) engine.
"""

import os
from PIL import Image, ImageDraw
from transformers import pipeline
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import numpy as np
import time

# path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Creating options for tesseract OCR engine
alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
options = "-c tessedit_char_whitelist={}".format(alphanumeric)
options += " --psm {}".format(7)


base_dir = os.path.dirname(__file__)
dataset_dir = "archive (2)"

if not os.path.isdir("detect_LP"):
    os.mkdir("detect_LP")
proc_img_dir = os.path.join(base_dir, "detect_LP")

image_dir = os.path.abspath(os.path.join(base_dir, dataset_dir))
images_pth = os.path.join(image_dir, os.listdir(image_dir)[-1])

images = os.listdir(images_pth)


for image in images:
    images[images.index(image)] = os.path.join(image_dir, "images", image)

def remove_border(licence_plate):
    (h, w) = licence_plate.shape

    # create markers to point the border lies at the edges or corner of image
    marker = np.zeros((h,w), dtype=np.uint8)
    for j in range(0, h):
        for k in range(0, w):
            if j == 0 or k == 0 or j == h-1 or k == w-1:
                marker[j,k] = licence_plate[j,k]

    # kernal for morphological dilation operation
    kernel = np.ones((5,5), dtype=np.uint8)

    while True:
        # dilate marker
        dilate_border = cv.dilate(marker, kernel)
        # mask the dilated image with the original roi image
        cv.bitwise_and(dilate_border, licence_plate, dilate_border)

        # if dilated markers becomee equal to the borders in image then break
        if (marker == dilate_border).all():
            break
        marker = dilate_border

    return licence_plate - dilate_border

# model to use
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")


# load input image  10
img = Image.open(images[110]) #cv.imread(images[0])

start = time.time()
prediction = detector(img, candidate_labels=["licence_plate"])[0]
end = time.time()

print(np.round(end-start, 2))

temp_image = np.array(img.copy())
channels = temp_image.shape[-1]

if channels == 4:
    temp_image = cv.cvtColor(temp_image, cv.COLOR_RGBA2RGB)
elif channels == 1:
    temp_image = cv.cvtColor(temp_image, cv.COLOR_GRAY2RGB)

LP_box = prediction['box']
xmin, ymin, xmax, ymax = LP_box.values()

cv.rectangle(temp_image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

# crop licence plate region in the input image for input to OCR
LP_roi = np.array(img.crop(list(LP_box.values())))
#LP_roi = cv.bilateralFilter(LP_roi, 12, 105,105)
LP_roi = cv.cvtColor(LP_roi, cv.COLOR_RGB2GRAY)

ROI = cv.threshold(LP_roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
roi = remove_border(ROI)

# recognize text in the LP using pytesseract OCR
LP_Text = pytesseract.image_to_string(roi, config=options)
LP_Text = LP_Text[:-1]
print("[licence plate] ", LP_Text)

cv.putText(temp_image, LP_Text, (xmin, ymin-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

fig, ax = plt.subplots(nrows=1, ncols=4)

ax[0].imshow(temp_image)
ax[1].imshow(LP_roi)
ax[2].imshow(ROI)
ax[3].imshow(roi)
plt.show()
