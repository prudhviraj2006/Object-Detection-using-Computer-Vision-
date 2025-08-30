import streamlit as st
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

# ------------------- App Title ------------------- #
st.title("🧠 Object Detection using Faster R-CNN")
st.write("Upload an image to detect objects with bounding boxes and confidence scores.")

# ------------------- Load Pre-trained Model ------------------- #
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

# ------------------- COCO Labels ------------------- #
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ------------------- Upload Image ------------------- #
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.01)

if uploaded_file:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # Draw predictions
        draw = ImageDraw.Draw(image)
        for idx in range(len(prediction["boxes"])):
            score = prediction["scores"][idx].item()
            if score >= threshold:
                box = prediction["boxes"][idx].tolist()
                label_idx = prediction["labels"][idx]
                label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                draw.rectangle(box, outline="lime", width=3)
                draw.text((box[0]+5, box[1]-10), f"{label} ({score:.2f})", fill="yellow")

        st.image(image, caption="Detected Objects", use_column_width=True)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
