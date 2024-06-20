# Extracting Images from PDF
#To extract images from a PDF, you can use libraries such as PyMuPDF, pdf2image, or PyPDF2. Here’s an example using PyMuPDF:
import fitz  # PyMuPDF

# Open the PDF
pdf_document = "D:/VSCODE/INTEREXT/cat/urban.pdf"
doc = fitz.open(pdf_document)

# Iterate through the pages
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    image_list = page.get_images(full=True)
    
    # Iterate through the images
    for image_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        # Save the image
        image_filename = f"page_{page_num + 1}_image_{image_index + 1}.png"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)

print("Images extracted successfully.")


#Analyzing Images
# Object Detection: YOLO, SSD, Faster R-CNN, EfficientDet
# Image Classification: ResNet, Inception, MobileNet, EfficientNet
# Image Segmentation: U-Net, Mask R-CNN, DeepLab
#You can use TensorFlow or PyTorch to load these models.

import tensorflow as tf
import cv2
import numpy as np

# Load the EfficientDet model
model = tf.saved_model.load('D:/VSCODE/INTEREXT/cat/efficientdet/efficientdet/model.py')

# Function to perform object detection
def detect_objects(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform detection
    detections = model(input_tensor)

    # Process detections
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detection_boxes = detections['detection_boxes']
    detection_classes = detections['detection_classes'].astype(np.int64)
    detection_scores = detections['detection_scores']

    return detection_boxes, detection_classes, detection_scores

# Example usage
image_path = "D:/VSCODE/INTEREXT/cat/01.png"
boxes, classes, scores = detect_objects(image_path)
print("Detected objects:", boxes, classes, scores)
