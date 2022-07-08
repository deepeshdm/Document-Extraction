"""
Send your Image as Base64 string at '/document_extract' and receive extracted document.
"""

from flask import Flask, request
import pybase64
import io
import cv2
import numpy as np
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


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

    im_file = io.BytesIO()
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


app = Flask(__name__)


@app.route('/document_extract', methods=["POST"])
def get_data():
    # process post request
    data = request.get_json()
    # extract the base64 string
    imageBase64 = data["imageBase64"]
    # decode base64 to numpy
    uploaded_image = decodeBase64ToImage(imageBase64)
    # Resize Image to (760x1020)
    img = cv2.resize(uploaded_image, (760, 1020))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)

    # --------------------------------------

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")
    # Build multipart form and post request
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

    # -------------------------------------------------------
    print("Detecting the document...")
    api_endpoint = "https://detect.roboflow.com/document-extraction-2/1?api_key=y3qRoLpRAP8xmsQWlEQf"
    response = requests.post(api_endpoint, data=m, headers={'Content-Type': m.content_type})

    if response.status_code != 200:
        return ("Detection Failed for some reason !")
    else:
        print("Detection complete...")
        detections = response.json()["predictions"]
        if len(detections) == 0:
            return ("No Document/Content Detected in given Image !")
        else:
            box = detections[0]
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2

            print("Cropping document part...")
            extracted_doc = pilImage.crop((x1, y1, x2, y2))

            # Encode output to base64
            imageBase64 = encodeImageToBase64(np.array(extracted_doc))
            return {"imageBase64": imageBase64}


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=1000, debug=True)
