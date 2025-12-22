import streamlit as st
from ai_detector import predict_proba, predict_label
import plotly.graph_objects as go
import io
from PyPDF2 import PdfReader
from docx import Document

# Session State Initialization
if "text" not in st.session_state:
    st.session_state["text"] = ""

# Indicator
def show_gauge(percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "white"},
            ],
        },
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=22),
    )

    st.plotly_chart(fig, use_container_width=True)

# Upload File and Extract Text
def read_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")

def read_pdf(uploaded_file) -> str:
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    pages_text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages_text.append(t)
    return "\n".join(pages_text)

def read_docx(uploaded_file) -> str:
    doc = Document(io.BytesIO(uploaded_file.read()))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    elif name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return read_docx(uploaded_file)
    return ""

# Ui
st.set_page_config(page_title="AI Detector", page_icon="🧠", layout="centered")

st.title("🧠 AI vs Human Text Detector")
st.write("Upload a document OR paste text and click **Detect**.")

# Upload section
uploaded_file = st.file_uploader(
    "Upload a .txt, .pdf, or .docx file",
    type=["txt", "pdf", "docx"]
)

if uploaded_file is not None:
    extracted = extract_text_from_upload(uploaded_file).strip()

    if not extracted:
        st.error("Couldn't extract text from this file. If it's a scanned PDF (image), text extraction won't work.")
    else:
        st.success(f"Extracted {len(extracted)} characters from: {uploaded_file.name}")
        # Auto-fill 
        st.session_state["text"] = extracted

# Main input box (uses session_state)
text = st.text_area(
    "Input text",
    height=220,
    key="text",
    placeholder="Paste paragraph here..."
)
# Buttons
col1, col2 = st.columns([1,1])
with col1:
    detect_clicked = st.button("Detect", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state["text"] = ""

# Detection
if detect_clicked:
    if not text.strip():
        st.warning("Please paste some text or upload a document.")
    else:
        label_int = predict_label(text)
        p_ai = predict_proba(text)

        label = "AI" if label_int == 1 else "Human"
        percent = p_ai * 100

        st.subheader(f"Result: {label}")
        show_gauge(percent)
        st.markdown("<p style='text-align:center; font-size:22px;'>AI Percentage</p>", unsafe_allow_html=True)
