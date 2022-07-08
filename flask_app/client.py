"""
A simple script to deal with the flask server.
"""

import pybase64
from PIL import Image
import io
import requests
import numpy as np
from io import BytesIO
import cv2


# Takes numpy array and decode's the image as base64
def encodeImageToBase64(numpy_img):
    """
    code referenced from :
    https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/

    Instead of saving the PIL Image object img to the disk, we save it to
    im_file which is a file-like object.
    """
    # Rescaling between 0-255 and convert to integers
    img = Image.fromarray(np.uint8(numpy_img * 255))
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_base64 = pybase64.standard_b64encode(im_bytes)
    # decode bytes to string
    encoded = im_base64.decode('utf-8')
    return encoded


# Takes base64 string and returns the Image as numpy array
def decodeBase64ToImage(base64_string):
    # decoding base54 to bytes
    print("Converting base64 to bytes...")
    decoded = pybase64.standard_b64decode(base64_string)
    print("LENGTH : ", len(decoded))
    print("TYPE1 : ", type(decoded))
    # converting bytes to image
    print("Converting bytes to image...")
    stream = io.BytesIO(decoded)
    img = Image.open(stream)
    print("Converting image to numpy array...")
    img = np.array(img)
    print("TYPE2 : ", type(img))
    return img


# Image Path
path = r"C:\Users\Deepesh\Desktop\training_demo\images\train\Image_16.jpeg"
img = cv2.imread(path)
# Resize Image to (760x1020)
img = cv2.resize(img, (760, 1020))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img)
# convert numpy to base64
imageBase64 = encodeImageToBase64(img)

# Send request to flask app
endpoint = "http://127.0.0.1:1000/document_extract"
response = requests.post(endpoint, json={"imageBase64": imageBase64})
# Decode base64 to numpy
imageBase64 = response.json()["imageBase64"]
img = decodeBase64ToImage(imageBase64)
# display image
cv2.imshow("output", img)
cv2.waitKey(0)
