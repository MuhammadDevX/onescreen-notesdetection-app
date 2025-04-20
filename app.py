import streamlit as st
from dotenv import load_dotenv
import os
import openai
import base64
import torch
from PIL import Image, ImageDraw  
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import io
import random


# --- Configuration ---
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Hugging Face model (no local download needed)
GROUNDING_DINO_HF = "IDEA-Research/grounding-dino-base"

system_prompt = """You are a helpful assistant that can detect whether an image can be used to create notes or not. You shall be provided with an image and you must respond with a yes or no followed by the description. If the image is not clear enough but you can detect any notes related material, you should respond with yes. Some examples for yes are:
- a whiteboard with notes on it
- a paper with notes on it
- a screenshot of a notes app
- a screenshot of a notes website
- a screenshot of a notes software
- a screenshot of a notes tool
- any graphical reresentation of notes
- a screenshot of a notes tool with notes on it 
etc

Some examples for no are:
- a whiteboard, paper, screenshot, or graphical representation without any notes on it
- a vague image that does not contain any notes related material
- a screenshot of a notes app without any notes on it
- a very low quality image that does not contain any notes related material"""


# --- Helper Functions ---
@st.cache_resource
def load_hf_model():
    """Load Hugging Face model (cached)"""
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_HF)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_HF)
    return processor, model


def detect_notes_hf(image_bytes, text_prompt="text. graph. handwriting. charts. drawings. diagrams. sketches. figures. images. equations. plane. geometry. notes. plots. flowcharts."):
    """Run HF Grounding DINO detection with debug output"""
    processor, model = load_hf_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Process inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs, 
        inputs.input_ids, 
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=target_sizes
    )[0]
    
    
     # Get raw boxes and scores
    raw_boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results["labels"]
    
    # Merge overlapping boxes
    merged_boxes = merge_boxes(raw_boxes)
    
    # Debug output
    print(f"\n📦 Box merging results:")
    print(f"Original {len(raw_boxes)} boxes → Merged to {len(merged_boxes)} boxes")
    
    # Draw merged boxes
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for box in merged_boxes:
        draw.rectangle(box, outline="red", width=3)
    
    return annotated_image, merged_boxes
  




def should_merge(box1, box2):
    """Check if two boxes overlap significantly"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    dx = min(x1_max, x2_max) - max(x1_min, x2_min)
    dy = min(y1_max, y2_max) - max(y1_min, y2_min)
    intersection = dx * dy if (dx > 0 and dy > 0) else 0
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    # Merge if overlap exceeds 30% of either box
    return (intersection / min(area1, area2)) > 0.3

def merge_boxes(boxes):
    """Merge overlapping boxes into single encompassing boxes"""
    if not boxes:
        return []
    
    # Convert to list of [x1, y1, x2, y2]
    boxes = [list(map(float, box)) for box in boxes]
    
    changed = True
    while changed:
        changed = False
        new_boxes = []
        merged = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if merged[i]:
                continue
                
            current_box = boxes[i]
            for j in range(i+1, len(boxes)):
                if not merged[j] and should_merge(current_box, boxes[j]):
                    # Merge boxes by taking max boundaries
                    current_box = [
                        min(current_box[0], boxes[j][0]),  # min x1
                        min(current_box[1], boxes[j][1]),  # min y1
                        max(current_box[2], boxes[j][2]),  # max x2
                        max(current_box[3], boxes[j][3])   # max y2
                    ]
                    merged[j] = True
                    changed = True
            
            new_boxes.append(current_box)
            merged[i] = True
        
        boxes = new_boxes
    
    return boxes

  

# --- Streamlit App ---
def main():
    st.title("📝 Cloud-Based Note Processor")
    st.markdown("""
    Uses:
    - GPT-4o-mini (API) for classification
    - HF Grounding DINO (cloud) for detection
    """)
    
    uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Images"):
        for uploaded_file in uploaded_files:
            with st.expander(f"Results for {uploaded_file.name}", expanded=True):
                # --- Classification ---
                image_bytes = uploaded_file.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode()
                data_url = f"data:image/{uploaded_file.type.split('/')[-1]};base64,{image_b64}"
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{
                      "role":"system",
                      "content":system_prompt
                      
                      },{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Is this image useful for note-making? Respond only 'yes' or 'no'."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }],
                )
                
                classification = response.choices[0].message.content.lower()
                
                # --- Detection ---
                if "yes" in classification:
                    st.success("✅ Good for notes - detecting regions...")
                    annotated_img, boxes = detect_notes_hf(image_bytes)
                    
                    # Display results
                    cols = st.columns(2)
                    cols[0].image(image_bytes, caption="Original")
                    cols[1].image(annotated_img, caption=f"Detected {len(boxes)} regions")
                    
                    # Export
                    buf = io.BytesIO()
                    annotated_img.save(buf, format="JPEG")
                    st.download_button(
                        "Download Annotated",
                        buf.getvalue(),
                        file_name=f"annotated_{uploaded_file.name}",
                        mime="image/jpeg",
                        key=f'{boxes}{len(boxes)*random.random()}'
                    )
                else:
                    st.error("❌ Not suitable for notes")
                
                st.divider()

if __name__ == "__main__":
    main()