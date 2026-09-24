import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import fitz  # pymupdf
import io
import base64

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Urdu OCR Editor",
    page_icon="📄",
    layout="wide"
)

# ─── CSS Styling ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
textarea {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    line-height: 2.8 !important;
    background-color: #fffdf5 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Gemini Client ──────────────────────────────────────────
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ─── Prompt ─────────────────────────────────────────────────
OCR_PROMPT = """
You are an expert OCR assistant specializing in Urdu and English text extraction.

You will receive an image of a scanned document that may contain:
- Urdu text (written in Nastaliq style, right to left)
- English text (left to right)
- Mixed Urdu and English on the same page

Your task:
1. Extract ALL text exactly as it appears in the image
2. Maintain the original structure and layout as much as possible
3. Keep Urdu text in proper Urdu Unicode characters
4. Keep English text as is
5. Do NOT translate anything
6. Do NOT add any extra text or explanation
7. Return only the extracted text, nothing else

Important:
- Preserve line breaks where visible
- If a word is unclear, write [unclear] in its place
"""

# ─── Helper: PDF to Image ───────────────────────────────────
def pdf_to_image(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

# ─── Helper: Image to bytes ─────────────────────────────────
def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# ─── Helper: Extract Text ───────────────────────────────────
def extract_text(image: Image.Image) -> str:
    img_bytes = image_to_bytes(image)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text=OCR_PROMPT),
        ]
    )
    return response.text

# ─── UI ─────────────────────────────────────────────────────
st.title("📄 Urdu / English OCR Editor")
st.caption("Scan شدہ دستاویز اپلوڈ کریں — قابلِ ترمیم متن حاصل کریں")

uploaded = st.file_uploader(
    "Image یا PDF اپلوڈ کریں",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded:
    # ─── File convert ───────────────────────────────────────
    if uploaded.type == "application/pdf":
        image = pdf_to_image(uploaded)
    else:
        image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 اپلوڈ شدہ دستاویز")
        st.image(image, use_container_width=True)   # ← FIX

    with col2:
        st.subheader("✏️ قابلِ ترمیم متن")

        font_size = st.slider("Font Size", min_value=14, max_value=36, value=20, step=2)

        st.markdown(f"""
        <style>
        textarea {{ font-size: {font_size}px !important; }}
        </style>
        """, unsafe_allow_html=True)

        if st.button("🔍 متن نکالیں (Extract Text)", use_container_width=True):
            with st.spinner("متن نکالا جا رہا ہے... براہ کرم انتظار کریں"):
                try:
                    extracted = extract_text(image)
                    st.session_state["extracted_text"] = extracted
                    st.success("متن کامیابی سے نکال لیا گیا ✅")
                except Exception as e:
                    st.error(f"خرابی آگئی: {str(e)}")

        if "extracted_text" in st.session_state:
            edited_text = st.text_area(
                "متن یہاں ترمیم کریں",
                value=st.session_state["extracted_text"],
                height=450,
                key="editor"
            )

            b64 = base64.b64encode(edited_text.encode()).decode()
            st.markdown(f"""
            <a href="data:text/plain;charset=utf-8;base64,{b64}" download="urdu_text.txt">
                <button style="
                    background:#4CAF50;
                    color:white;
                    border:none;
                    padding:10px 20px;
                    border-radius:5px;
                    cursor:pointer;
                    font-size:16px;
                    width:100%;
                    margin-top:10px;">
                    💾 Text Download کریں
                </button>
            </a>
            """, unsafe_allow_html=True)
