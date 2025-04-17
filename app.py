import streamlit as st
from dotenv import load_dotenv
import os
import openai
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

# System prompt (same as in notebook)
system_prompt = """You are a helpful assistant that can detect whether an image can be used to create notes or not. You shall be provided with an image and you must respond with a simple yes or no. If the image is not clear enough but you can detect any notes related material, you should respond with yes. You should not provide any other information or explanation. Some examples for yes are:
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
- a very low quality image that does not contain any notes related material"""  # (keep your full system prompt here)

def process_uploaded_images(uploaded_files):
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
            # Read image data directly from uploaded file
            image_bytes = uploaded_file.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode()
            data_url = f"data:image/{uploaded_file.type.split('/')[-1]};base64,{image_b64}"
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Is this image useful for note making: {uploaded_file.name}?"},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]
            )
            
            response_text = response.choices[0].message.content.lower()
            is_good = "yes" in response_text
            
            results.append({
                "name": uploaded_file.name,
                "bytes": image_bytes,
                "response": response_text,
                "is_good": is_good
            })
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    st.title("📄 ONESCREEN Note-Worthy Image Classifier")
    st.markdown("""
    Upload images to classify them for note-making suitability.
    - ✅ Green border: Good for notes
    - ❌ Red border: Not suitable for notes
    """)
    
    # File uploader with drag-and-drop support
    uploaded_files = st.file_uploader(
        "Choose images to analyze",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Analyze Images", type="primary"):
            results = process_uploaded_images(uploaded_files)
            
            good_count = sum(1 for r in results if r["is_good"])
            bad_count = len(results) - good_count
            
            # Summary message
            if bad_count == 0:
                st.success(f"🎉 All {len(results)} images are suitable for note-making!")
                st.balloons()
            else:
                st.info(f"Results: {good_count} ✅ suitable | {bad_count} ❌ not suitable")
            
            # Display all images with classification
            cols = st.columns(2)
            col_index = 0
            
            for result in results:
                with cols[col_index]:
                    # Create a container with colored border
                    border_color = "green" if result["is_good"] else "red"
                    border_style = f"solid 4px {border_color}"
                    
                    container = st.container(border=True)
                    with container:
                        # Apply border style
                        st.markdown(
                            f'<div style="border: {border_style}; border-radius: 8px; padding: 10px;">',
                            unsafe_allow_html=True
                        )
                        
                        # Display image
                        st.image(result["bytes"], use_container_width=True)
                        
                        # Display filename and response
                        emoji = "✅" if result["is_good"] else "❌"
                        st.markdown(f"""
                        **{emoji} {result["name"]}**  
                        **AI Response:** {result["response"]}
                        """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Alternate between columns
                col_index = (col_index + 1) % 2

if __name__ == "__main__":
    main()