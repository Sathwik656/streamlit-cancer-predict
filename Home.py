import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🩺",
    layout="wide"
)

with st.sidebar:
    st.markdown("## 🩺 Breast Cancer App")
    st.caption("ML-assisted cytology analysis")

    st.divider()

    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("🔬 Cancer Predictor", use_container_width=True):
        st.switch_page("pages/Predictor.py")

    st.divider()

    st.caption(
        "This tool assists in analyzing cell nuclei measurements "
        "using a trained machine learning model."
    )


st.title("Breast Cancer Detection System")
st.subheader("Machine Learning–Assisted Cytology Analysis")

st.markdown(
    """
    This application is designed to assist in the **early assessment of breast cancer**
    using **cell nuclei measurements** derived from cytology samples.

    A trained **machine learning model** analyzes morphological features of breast cell
    clusters and estimates whether the tissue is **benign or malignant**.
    """
)

st.divider()

st.header("How This Application Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Input**
        - Cell nuclei measurements from cytology samples
        - Includes radius, texture, perimeter, area, and shape features
        - Values can be adjusted manually using sliders
        """
    )

with col2:
    st.markdown(
        """
        **Output**
        - Predicted classification (Benign / Malignant)
        - Probability scores for each outcome
        - Radar chart visualization of feature distributions
        """
    )

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This application is intended for educational and research purposes only.
    It should **not** be used as a substitute for professional medical diagnosis
    or clinical decision-making.
    """
)

st.divider()

st.header("Get Started")

st.markdown(
    """
    Click the button below to proceed to the **Breast Cancer Predictor**
    and begin analyzing cytology measurements.
    """
)

if st.button("Go to Cancer Predictor", use_container_width=True):
    st.switch_page("pages/Predictor.py")
