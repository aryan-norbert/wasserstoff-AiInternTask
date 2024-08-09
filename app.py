%%writefile streamlit_app/app.py

import streamlit as st
import os
from PIL import Image
import cv2
import easyocr
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load YOLO model
model_yolo = YOLO('yolov8n.pt')  # Replace with your YOLO model path

# Load captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize OCR reader
reader = easyocr.Reader(['en'])  # You can specify other languages as needed

def identify_and_segment(image_path, output_dir):
    img = cv2.imread(image_path)
    results = model_yolo(image_path, save=False)
    table_data = []
    seq_number = 1
    for idx, det in enumerate(results[0].boxes.data):
        class_id = int(det[5])
        class_name = model_yolo.names[class_id]
        xmin, ymin, xmax, ymax, conf, _ = det
        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        output_filename = os.path.join(output_dir, f"{class_name}{idx+1}.jpg")
        cv2.imwrite(output_filename, cropped_img)

        # Generate caption for the segment
        raw_image = Image.open(output_filename).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model_caption.generate(**inputs)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0]

        table_data.append([seq_number, class_name, caption])
        seq_number += 1
    return table_data

def extract_text(image_path):
    result = reader.readtext(image_path)
    extracted_text = "\n".join([detection[1] for detection in result])
    return extracted_text

def main():
    st.title("Image Analysis App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(image_path)

        # Identify and segment objects
        segmented_dir = "temp_segmented"
        os.makedirs(segmented_dir, exist_ok=True)
        table_data = identify_and_segment(image_path, segmented_dir)

        # Display segmented images and table
        st.subheader("Segmented Objects and Descriptions")
        for row in table_data:
            col1, col2 = st.columns(2)
            segment_path = os.path.join(segmented_dir, f"{row[1]}{row[0]}.jpg")
            col1.image(Image.open(segment_path), use_column_width=True)
            col2.write(f"**{row[1]}**: {row[2]}")

        # Extract text from image
        extracted_text = extract_text(image_path)
        st.subheader("Extracted Text")
        st.write(extracted_text)

        # Placeholder for image summary (you'll need to implement this)
        st.subheader("Image Summary")
        st.write("This feature is coming soon!")

if __name__ == "__main__":
    main()
