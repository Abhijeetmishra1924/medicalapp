# streamlit_app.py

import streamlit as st
from openai import OpenAI
from PIL import Image
import io

st.set_page_config(page_title="AI Medical Image Explainer", layout="centered")

st.title("🩺 AI Medical Image Explainer")
st.write("Upload a medical image (X-ray, MRI, etc.), and get an easy-to-understand explanation of abnormalities.")

# Initialize OpenAI-compatible client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

# Model: Google Gemini 1.5 Flash via OpenRouter
MODEL = "google/gemini-1.5-flash"

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    with col2:
        with st.spinner("Analyzing image with AI..."):
            try:
                # Convert image to bytes
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()

                # Call OpenRouter (Gemini Flash)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Explain any abnormalities in this medical image in simple, non-technical terms suitable for patients or students."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=500,
                )
                explanation = response.choices[0].message.content
            except Exception as e:
                st.error(f"Error: {e}")
                explanation = None

    if explanation:
        with st.expander("📄 AI Explanation", expanded=True):
            st.write(explanation)

        # Download as text
        st.download_button(
            label="💾 Download Explanation as Text",
            data=explanation,
            file_name="medical_image_explanation.txt",
            mime="text/plain"
        )
