import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import sys


# PATH SETUP
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "model", "scaler.pkl")
CSS_PATH = os.path.join(ROOT_DIR, "assets", "styles.css")

def get_clean_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),

        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),

        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_value(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        min_val = X[key].min()
        max_val = X[key].max()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)

    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_value(input_data)

    categories = [
        'Radius', 'Texture', 'Perimeter', 'Area',
        'Smoothness', 'Compactness', 'Concavity',
        'Concave Points', 'Symmetry', 'Fractal Dimension'
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'],
            input_data['perimeter_mean'], input_data['area_mean'],
            input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'],
            input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'],
            input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'],
            input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'],
            input_data['perimeter_worst'], input_data['area_worst'],
            input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'],
            input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='black',
                linecolor='black',
                tickcolor='black'
            ),
            angularaxis=dict(
                gridcolor='black',
                linecolor='black',
                tickcolor='black'
            )
        ),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))

    data = get_clean_data()
    feature_names = data.drop(['diagnosis'], axis=1).columns

    input_array = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prob_malignant = model.predict_proba(input_array_scaled)[0][1]
    prob_benign = model.predict_proba(input_array_scaled)[0][0]

    st.subheader("Cell cluster prediction")

    if prob_malignant < 0.5:
        st.markdown("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malicius'>Malicious</span>", unsafe_allow_html=True)

    st.write(f"Probability of being benign: {prob_benign:.4f}")
    st.write(f"Probability of being malignant: {prob_malignant:.4f}")

    st.caption(
        "This app can assist medical professionals in making a diagnosis, "
        "but should not be used as a substitute for professional medical advice."
    )

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    input_data = add_sidebar()

    st.title("Breast Cancer Predictor")
    st.write(
        "This page predicts whether a breast mass is benign or malignant "
        "based on cytology measurements using a trained machine learning model."
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        st.plotly_chart(get_radar_chart(input_data), use_container_width=True)

    with col2:
        add_predictions(input_data)

main()
