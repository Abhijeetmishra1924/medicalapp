import streamlit as st
from PIL import Image
import io
import base64
import requests
import json

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="AI Medical Image Explainer",
    page_icon="🩺",
    layout="wide"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
}
.title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2a9d8f;
    text-align: center;
}
.subtitle {
    font-size: 1.2rem;
    color: #264653;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown('<h1 class="title">AI Medical Image Explainer 🩺</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload X-rays or MRIs and get easy-to-understand analysis</p>', unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.header("Upload Medical Image")
uploaded_file = st.sidebar.file_uploader("Choose an X-ray or MRI", type=["png", "jpg", "jpeg"])
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Abhijeet Mishra**")
st.sidebar.markdown("Powered by Google Gemini 2.5 API via OpenRouter")

# ------------------ Main App ------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.info("Analyzing image... This may take a few seconds.")

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Prepare API request
    api_key = st.secrets["OPENROUTER_API_KEY"]
    endpoint = "https://api.openrouter.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "google/gemini-2.5-flash-image-preview:free",
        "messages": [
            {
                "role": "user",
                "content": "Explain any abnormalities in this medical image in simple terms."
            }
        ],
        "image": img_str  # send image as separate field
    }

    # Make API request
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            # Updated path depending on OpenRouter response
            explanation = ""
            if "output_text" in result:
                explanation = result["output_text"]
            elif "choices" in result and len(result["choices"]) > 0:
                explanation = result["choices"][0].get("message", {}).get("content", "")
            else:
                explanation = json.dumps(result, indent=2)

            st.success("Analysis Complete ✅")
            with st.expander("Show Explanation"):
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
