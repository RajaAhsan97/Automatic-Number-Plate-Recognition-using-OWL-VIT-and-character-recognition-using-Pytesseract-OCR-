"""
    This code utilized conventional image processing techniques to recognition license plate in the input image and processing it further 
    to recognize text using Pytesseract OCR engine.
"""

import os
import numpy as np
import imutils
import pytesseract
from skimage.segmentation import clear_border
import cv2 as cv

# path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Creating options for tesseract OCR engine
alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
options = "-c tessedit_char_whitelist={}".format(alphanumeric)
options += " --psm {}".format(7)


base_dir = os.path.dirname(__file__)
dataset_fldr = "archive (2)"

image_dir = os.path.abspath(os.path.join(base_dir, dataset_fldr))
images_pth = os.path.join(image_dir, os.listdir(image_dir)[-1])

images = os.listdir(images_pth)
for image in images:
    images[images.index(image)] = os.path.join(image_dir, "images", image)

if not os.path.isdir("processed_images"):
    os.mkdir("processed_images")
save_path = os.path.join(base_dir, "processed_images")


def ANRP_image_preprocessing(gray, pth):
    """
        The dataset for ANPR contains images of vehicles with international
        number plates. Therefore for prior recognition of number plate in the
        given input image frame, a rectangular kernel having size (15,7) is
        initialized and applying morphological operation on the image to detect
        region of licence plate. The licence plate contains dark characters on a
        ligher background region.
    """
    # perform morphological operation on input image in order to reveal the
    # dark regions (i.e. text) on the lighter background
    # create rectangular shape kernel for morphological operation
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (13,5))
    img_morph = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rect_kernel)
    cv.imwrite(os.path.join(pth, "blackhat.jpg"),img_morph)

    # finding lighter region in the image
    sqr_kernel =  cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    light = cv.morphologyEx(gray, cv.MORPH_CLOSE, sqr_kernel)
    light = cv.threshold(light, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    cv.imwrite(os.path.join(pth, "light_reg.jpg"),img_morph)

    # finding gradient in the image
    gradient_x = np.absolute(cv.Sobel(img_morph, ddepth=cv.CV_32F, dx=1, dy=0, ksize=1))
    minVal, maxVal = np.min(gradient_x), np.max(gradient_x)
    # normalizing gradient
    gradient_x = 255 * ((gradient_x - minVal)/(maxVal - minVal))
    gradient_x = gradient_x.astype("uint8")
    cv.imwrite(os.path.join(pth, "gradient.jpg"),img_morph)

    # smoothing boundary regions
    gradient_x = cv.GaussianBlur(gradient_x, (5,5), 0)
    gradient_x = cv.morphologyEx(gradient_x, cv.MORPH_CLOSE, rect_kernel)
    thresh = cv.threshold(gradient_x, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    cv.imwrite(os.path.join(pth, "gradient_thresh.jpg"),img_morph)

    # perform erosion and dilation morphology operation to clean the thresholded image
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=5)
    cv.imwrite(os.path.join(pth, "thresh_erode_dilate.jpg"),img_morph)

    thresh = cv.bitwise_and(thresh, thresh, mask=light)
    thresh = cv.dilate(thresh, None, iterations=2)
    thresh = cv.erode(thresh, None, iterations=1)
    cv.imwrite(os.path.join(pth, "thresh_final.jpg"),img_morph)

    return thresh


def find_contours(thresh, gray_img, min_ar, max_ar, pth):
    """
        Once the image is preprocessed, next step is to find the contours in the
        image and filtering the contour of licence plate by aspect ratio.
    """
    # determine contour in the thresholded image 
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # sorting contours in descending order (larger size contour to small size contour)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    LP_contour, ROI = None, None
    # load each detected contour and calculate aspect ratio 
    for contour in contours:
        (x,y,w,h) = cv.boundingRect(contour)
        ar = w/float(h)
        print(ar)


        # check if aspect ratio is rectangular
        if ar >= min_ar and ar <= max_ar:
            LP_contour = contour
            licencePlate = gray_img[y:y+h, x:x+w]
            ROI = cv.threshold(licencePlate, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            # cleaning the border of licence plate ROI
            ROI = clear_border(ROI)
            cv.imwrite(os.path.join(pth, "roi.jpg"), ROI)

    return ROI, LP_contour

def Apply_OCR(lp_roi, options):
    return pytesseract.image_to_string(lp_roi, config=options)

print("Processing input frames for ANPR...")
for image in images:
    img_nm = image.split("\\")[-1].split(".")[0]
    path = os.path.join(save_path, img_nm)
    if not os.path.isdir(path):
        os.mkdir(path)

    # read input image
    sample_img = cv.imread(image)
    cv.imwrite(os.path.join(path, "input_image.jpg"), sample_img)
    
    channels = sample_img.shape[-1]

    # convert image color channels to gray scale
    if channels == 3:
        gray = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY)
    elif channels == 4:
        gray = cv.cvtColor(sample_img, cv.COLOR_BGRA2GRAY)

    threshold = ANRP_image_preprocessing(gray, path)

    min_ar, max_ar = 3, 25
    roi, lpContour = find_contours(threshold, gray, min_ar, max_ar, path)

    # grab licence plate text using pytesserct OCR engine
    if roi is not None:
        LP_Text = Apply_OCR(roi, options)
        print("[Licence-plate]", LP_Text)
    else:
        print("Unable to detect licence plate on the preprocessed image")

    if LP_Text is not None and lpContour is not None:
        box = cv.boxPoints(cv.minAreaRect(lpContour)).astype('int')

        cv.drawContours(sample_img, [box], -1, (0,255,0), 2)

        (x,y,w,h) = cv.boundingRect(lpContour)
        cv.putText(sample_img, LP_Text[:-1], (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        cv.imwrite(os.path.join(path, "result.jpg"), sample_img)
