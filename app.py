import streamlit as st
from ai_detector import predict_proba, predict_label
import plotly.graph_objects as go

def show_gauge(percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "bgcolor": "white",
            "steps": [{"range": [0, 100], "color": "#f2f2f2"}],
        },
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit App
st.set_page_config(page_title="AI Detector", page_icon="🧠", layout="centered")

st.title("🧠 AI vs Human Text Detector")
st.write("Paste a paragraph and click **Detect**.")

text = st.text_area("Input text", height=220, placeholder="Paste paragraph here...")

if st.button("Detect"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        label_int = predict_label(text)       # 0 or 1
        p_ai = predict_proba(text)            # 0.0 - 1.0

        label = "AI" if label_int == 1 else "Human"
        percent = p_ai * 100
        show_gauge(percent)

        st.markdown("<p style='text-align:center; font-size:22px;'>AI Percentage</p>", unsafe_allow_html=True)

