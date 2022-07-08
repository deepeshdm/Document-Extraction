
# NOTE : You may only need to change the Image path, rest all paths are self defined.

# Path to Input Image
IMAGE_PATH = r"dataset/Image_86.jpeg"

# Path to trained model
PATH_TO_SAVED_MODEL = r"model/saved_model"

# Path to labelmap file
PATH_TO_LABELS = r"model/label_map.pbtxt"

# Decision Threshold - 65%
MIN_CONF_THRESH = float(0.65)

# ----------------------Load Model-------------------------------------------------

import tensorflow as tf
tf.gfile = tf.io.gfile
import cv2
import time
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load the trained model
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(round(elapsed_time, 3)))

# Load LabelMap data for plotting
category_index = label_map_util. \
    create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# ------------------------Perform Detection-----------------------------------------------

print('Running inference for {}... '.format(IMAGE_PATH), end='')

# Load the input image
image = cv2.imread(IMAGE_PATH)
# resize to (760,1020)
image = cv2.resize(image, (760, 1020))
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# Perform object detection
detections = detect_fn(input_tensor)
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections
# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

# ------------------------Display Output-----------------------------------------------

# Draw Bounding Boxes and Display it.
image_with_detections = image.copy()
print(detections['detection_classes'][:10])
print(detections['detection_scores'][:10])
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=MIN_CONF_THRESH,
    agnostic_mode=False)
print('Done')
cv2.imshow("output", image_with_detections)
cv2.waitKey(0)
