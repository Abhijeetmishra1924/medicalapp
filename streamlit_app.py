import streamlit as st
import requests
import json
from PIL import Image
import io

st.set_page_config(page_title="AI Medical Image Explainer", page_icon="🩺")

st.title("AI Medical Image Explainer 🩺")
uploaded_file = st.file_uploader("Upload X-ray or MRI", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert uploaded image to bytes and upload to a temporary image hosting service
    # (OpenRouter requires a public URL for image_url)
    # For demo, you can use imgbb, imgur, or any public URL
    st.info("Please make sure the image has a public URL for analysis.")
    image_url = st.text_input("Enter public image URL (required):")

    if image_url:
        api_key = st.secrets["OPENROUTER_API_KEY"]
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
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
                st.write(explanation)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Enter a public image URL for analysis.")
