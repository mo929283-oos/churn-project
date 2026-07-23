
import joblib
import streamlit as st

from config import (
    MODEL_PATH,
    SCALER_PATH,
    MODEL_COLUMNS_PATH
)


@st.cache_resource(show_spinner=False)
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_resource(show_spinner=False)
def load_scaler():

    scaler = joblib.load(SCALER_PATH)

    return scaler


@st.cache_resource(show_spinner=False)
def load_columns():

    columns = joblib.load(MODEL_COLUMNS_PATH)

    return columns


@st.cache_resource(show_spinner=False)
def load_all():

    model = load_model()

    scaler = load_scaler()

    columns = load_columns()

    return model, scaler, columns


def validate():

    errors = []

    try:
        load_model()
    except Exception as e:
        errors.append(
            f"Model Error : {e}"
        )

    try:
        load_scaler()
    except Exception as e:
        errors.append(
            f"Scaler Error : {e}"
        )

    try:
        load_columns()
    except Exception as e:
        errors.append(
            f"Columns Error : {e}"
        )

    return errors
