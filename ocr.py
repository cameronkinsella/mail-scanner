import os

import cv2
import imutils
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from cnn import Net
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keys.json'


def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# Load image, grayscale, Otsu's threshold
def get_chars(path_name):
    gcs = storage.Client()

    bucket = gcs.get_bucket('mail-scanner-bucket')
    blob = bucket.blob(path_name)
    blob.download_to_filename('image.jpg')
    image = cv2.imread('image.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

    warped = four_point_transform(image, screenCnt.reshape(4, 2) * (image.shape[0] / 480.0))

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 17, 4)


    image = imutils.resize(warped, width=500)

    # Remove Salt and pepper noise
    saltpep = cv2.fastNlMeansDenoising(image, None, 9, 13)

    # blur
    blured = cv2.blur(saltpep, (3, 3))

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    matrix = [[], [], []]

    for i, ctr in enumerate(sorted_ctrs):

        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        im = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        ret_1, thresh_1 = cv2.threshold(im,  127, 255, cv2.THRESH_BINARY_INV)
        im, ctrs_1, hier = cv2.findContours(thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        sorted_ctrs_1 = sorted(ctrs_1, key=lambda ctr: cv2.boundingRect(ctr)[0])
        count, p_height = 0, 0
        for j, ctr_1 in enumerate(sorted_ctrs_1):
            # Get bounding box
            x_1, y_1, w_1, h_1 = cv2.boundingRect(ctr_1)
            if w_1 > 50 and h_1 > 50:
                # Getting ROI
                roi_1 = thresh_1[y_1:y_1 + h_1, x_1:x_1 + w_1]

                if j != 0:
                    # print(h_1, p_height)
                    if p_height < h_1:
                        count = 0
                    matrix[count].append(roi_1)
                    p_height = h_1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net()
    model.load_state_dict(torch.load('letter_model.pt', map_location=torch.device(device)))

    p = transforms.Compose([transforms.Resize((28, 28)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])

    real_pred = ''
    for img in matrix[0]:
        img = Image.fromarray(img)
        img = p(img)
        img = img.view(1, 28, 28)
        img = img.unsqueeze(0)

        img = model(img)
        prediction = list(img.cpu().detach().numpy()[0])
        classes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        max_pred = prediction.index(max(prediction))
        real_pred += classes[max_pred]

    return real_pred

