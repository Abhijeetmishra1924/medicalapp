import streamlit as st
from PIL import Image
import io
import requests
import json

# ------------------ Page Config ------------------
st.set_page_config(page_title="AI Medical Image Explainer", page_icon="🩺", layout="wide")
st.title("AI Medical Image Explainer 🩺")
st.write("Upload an X-ray or MRI and get AI analysis in simple terms.")

# ------------------ Image Upload ------------------
uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("Uploading image for analysis...")

    # ------------------ Upload to Imgbb ------------------
    imgbb_api_key = st.secrets["IMGBB_API_KEY"]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    try:
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": imgbb_api_key},
            files={"image": img_bytes}
        )
        result = response.json()
        if result["success"]:
            image_url = result["data"]["url"]
            st.success("Image uploaded successfully!")
        else:
            st.error("Failed to upload image.")
            st.stop()
    except Exception as e:
        st.error(f"Error uploading image: {e}")
        st.stop()

    # ------------------ Send to OpenRouter Gemini 2.5 ------------------
    st.info("Analyzing image... This may take a few seconds.")
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "google/gemini-2.5-flash-image-preview:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this medical image and explain abnormalities in simple terms."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            explanation = result.get("output_text", "")
            st.success("Analysis Complete ✅")
            st.subheader("Explanation:")
            st.write(explanation)

            # Download Explanation as TXT
            st.download_button(
                label="Download Explanation",
                data=explanation,
                file_name="medical_image_analysis.txt",
                mime="text/plain"
            )
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.warning("Please upload a medical image to analyze.")
