import streamlit as st
from PIL import Image
import io
import base64
from groq import Groq

st.set_page_config(page_title="AI Medical Image Explainer (Groq)", page_icon="🩺")
st.title("AI Medical Image Explainer 🩺")
uploaded_file = st.file_uploader("Upload X-ray or MRI", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # ----------------- Groq API call -----------------
    st.info("Analyzing image...")
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    message_text = f"Analyze this medical image and explain abnormalities in simple terms. Here is the image in base64: {img_str}"

    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role":"user", "content": message_text}],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        reasoning_effort="default",
        stream=True
    )

    explanation = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            explanation += content
            st.write(content, end="")  # real-time streaming

    st.success("Analysis Complete ✅")
    st.download_button(
        label="Download Explanation",
        data=explanation,
        file_name="medical_image_analysis.txt",
        mime="text/plain"
    )
else:
    st.warning("Please upload a medical image to analyze.")
