from openai import OpenAI
import streamlit as st
from PIL import Image
import base64
import io

st.title("🩺 AI Medical Image Explainer")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("Analyze"):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        with st.spinner("Getting explanation..."):
            try:
                response = client.chat.completions.create(
                    model="google/gemini-1.5-flash",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Explain any abnormalities in simple terms."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")
