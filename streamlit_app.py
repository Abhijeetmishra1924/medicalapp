import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io

st.set_page_config(page_title="🩺 AI Medical Image Explainer", layout="centered")
st.title("🩺 AI Medical Image Explainer")
st.write("Upload a medical image (X-ray, MRI, etc.), and get an easy-to-understand explanation.")

# ✅ Load API key from secrets
api_key = st.secrets["OPENROUTER_API_KEY"]

# ✅ Initialize client safely
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "https://your-app-name.streamlit.app",  # Replace with your actual URL
        "X-Title": "AI Medical Image Explainer"
    }
)

MODEL = "google/gemini-1.5-flash"

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Analyze Image"):
        with st.spinner("Asking AI to explain..."):
            try:
                # Convert image to base64
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode()

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
                                        "url": f"data:image/jpeg;base64,{img_b64}"
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=500,
                )
                explanation = response.choices[0].message.content

                with st.expander("📄 AI Explanation", expanded=True):
                    st.write(explanation)

                st.download_button(
                    label="💾 Download Explanation",
                    data=explanation,
                    file_name="medical_explanation.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
