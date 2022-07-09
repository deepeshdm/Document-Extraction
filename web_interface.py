import streamlit as st
import numpy as np
import io
import cv2
import requests
from PIL import Image, ImageFont, ImageDraw
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Set page configs.
st.set_page_config(page_title="Document Extraction", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Document Detection & Extraction </p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "<div align='center'> Upload your Image and our ML "
    "model will detect the document and extract it for you </div>", unsafe_allow_html=True)

# -------------Upload Section------------------------------------------------

# Example Images
st.image(image="assets/example.png")
st.markdown("</br>", unsafe_allow_html=True)

# Upload the Image
content_image = st.file_uploader(
    "Image must be of size (760x1020) for optimal result. Images bigger than (760x1020) will be resized, resulting in reduced image quality.",
    type=['png', 'jpg', 'jpeg'])

st.markdown("</br>", unsafe_allow_html=True)

if content_image is not None:

    with st.spinner("Scanning the Image...will take about 10-15 secs"):

        #-------------------------------------------------------
        upload_image = Image.open(content_image)
        upload_image = np.array(upload_image)

        # Resize Image to (760x1020)
        img = cv2.resize(upload_image, (760, 1020))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(image)

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
            st.markdown(
                "<div align='center'> Detection Failed for some reason ! </div>", unsafe_allow_html=True)
        else:
            print("Detection complete...")
            detections = response.json()["predictions"]

            if len(detections) == 0:
                print("No Document/Content Detected in given Image !")
                st.markdown(
                    "<div align='center'> No Document or Content Detected ! </div>", unsafe_allow_html=True)
            else:
                box = detections[0]
                x1 = box['x'] - box['width'] / 2
                x2 = box['x'] + box['width'] / 2
                y1 = box['y'] - box['height'] / 2
                y2 = box['y'] + box['height'] / 2

                # Display the output with bbox
                image_with_detections = pilImage.copy()
                draw = ImageDraw.Draw(image_with_detections)
                font = ImageFont.load_default()
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                image_with_detections = cv2.cvtColor(np.array(image_with_detections), cv2.COLOR_BGR2RGB)
                st.image(image_with_detections)

                print("Cropping document part...")
                extracted_doc = pilImage.crop((x1, y1, x2, y2))
                # convert BGR to RGB
                extracted_doc = np.array(extracted_doc)
                extracted_doc = cv2.cvtColor(extracted_doc, cv2.COLOR_BGR2RGB)

                # Download option
                buffered = io.BytesIO()
                extracted_doc.save(buffered, format="JPEG")
                st.download_button(
                    label="Download Document",
                    data=buffered.getvalue(),
                    file_name="output.png",
                    mime="image/png")
