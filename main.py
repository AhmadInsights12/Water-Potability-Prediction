"""Streamlit front end for the water-potability prediction project."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from water_backend import (
    DATASET_PATH,
    FEATURE_LABELS,
    load_model,
    predict_samples,
)


st.set_page_config(
    page_title="ClearWater | Potability insights",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset was not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


st.sidebar.title("ClearWater")
st.sidebar.caption("Water potability analysis")
page = st.sidebar.radio(
    "Choose a page",
    ["Overview", "Single sample", "Batch predictions"],
    label_visibility="collapsed",
)

try:
    model_bundle = get_model()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(
        f"Unable to load the water prediction model: {error}",
        icon=":material/error:",
    )
    st.stop()

st.title(page)
st.caption(
    "A machine-learning demonstration using the water potability dataset."
)

if page == "Overview":
    try:
        dataset = get_dataset()
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        st.error(f"Unable to load the project dataset: {error}")
        st.stop()

    if "Potability" not in dataset:
        st.error("The project dataset is missing its Potability target column.")
        st.stop()

    total_samples = len(dataset)
    potable_count = int((dataset["Potability"] == 1).sum())
    not_potable_count = int((dataset["Potability"] == 0).sum())
    missing_measurements = int(
        dataset.reindex(columns=model_bundle.features).isna().sum().sum()
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Dataset samples", f"{total_samples:,}", border=True)
    metric_columns[1].metric("Potable samples", f"{potable_count:,}", border=True)
    metric_columns[2].metric(
        "Not-potable samples", f"{not_potable_count:,}", border=True
    )
    metric_columns[3].metric(
        "Missing measurements", f"{missing_measurements:,}", border=True
    )

    chart_column, details_column = st.columns([1, 1])
    with chart_column:
        st.subheader("Dataset class balance")
        class_balance = pd.DataFrame(
            {
                "Classification": ["Not potable", "Potable"],
                "Samples": [not_potable_count, potable_count],
            }
        )
        st.bar_chart(
            class_balance,
            x="Classification",
            y="Samples",
            alt="Number of potable and not-potable samples in the dataset",
        )
    with details_column:
        st.subheader("Model inputs")
        st.caption(
            f"{len(model_bundle.features)} measurements expected by the "
            f"{type(model_bundle.estimator).__name__} classifier."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Measurement": [
                        FEATURE_LABELS.get(feature, feature)
                        for feature in model_bundle.features
                    ],
                    "Missing-value default": [
                        model_bundle.defaults[feature]
                        for feature in model_bundle.features
                    ],
                }
            ),
            hide_index=True,
            alt="Model input measurements and defaults used for missing values",
        )

    st.info(
        "This model's output is an educational estimate, not a laboratory test "
        "or a guarantee that water is safe to drink.",
        icon=":material/info:",
    )

elif page == "Single sample":
    st.write("Enter the water measurements and submit to get a model estimate.")
    with st.form("single_prediction_form"):
        columns = st.columns(3)
        values: dict[str, float] = {}
        for index, feature in enumerate(model_bundle.features):
            label = FEATURE_LABELS.get(feature, feature)
            maximum = 14.0 if feature == "ph" else None
            values[feature] = columns[index % 3].number_input(
                label,
                min_value=0.0,
                max_value=maximum,
                value=float(model_bundle.defaults[feature]),
                step=0.1,
                format="%.4f",
                help=(
                    "Enter a value between 0 and 14."
                    if feature == "ph"
                    else "Enter a non-negative measurement."
                ),
            )
        submitted = st.form_submit_button(
            "Run prediction",
            type="primary",
            icon=":material/science:",
            width="stretch",
        )

    if submitted:
        prediction = predict_samples(pd.DataFrame([values]), model_bundle).iloc[0]
        probability = prediction["potable_probability"]
        if prediction["predicted_potability"] == "Potable":
            st.success(
                "The model predicts this sample as potable.",
                title="Predicted potable",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                "The model predicts this sample as not potable.",
                title="Predicted not potable",
                icon=":material/water_damage:",
            )

        if pd.notna(probability):
            st.metric("Estimated potable probability", f"{probability:.1%}")
        st.caption(
            "This estimate is not a substitute for certified water-quality testing."
        )

else:
    st.write(
        "Upload a CSV with one column for each model measurement. "
        "Extra columns are ignored; blank measurements use the model's "
        "training-set median defaults."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="The project water_potability.csv is also accepted.",
    )

    if uploaded_file is not None:
        try:
            samples = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
        else:
            try:
                results = predict_samples(samples, model_bundle)
            except (TypeError, ValueError, RuntimeError) as error:
                st.error(f"Could not predict this file: {error}")
            else:
                potable = int((results["predicted_potability"] == "Potable").sum())
                not_potable = len(results) - potable
                first, second, third = st.columns(3)
                first.metric("Rows processed", f"{len(results):,}", border=True)
                second.metric("Predicted potable", f"{potable:,}", border=True)
                third.metric("Predicted not potable", f"{not_potable:,}", border=True)

                st.dataframe(
                    results,
                    hide_index=True,
                    alt="Uploaded water measurements and model predictions",
                )
                st.download_button(
                    "Download predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="water_potability_predictions.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    width="stretch",
                )
                st.caption(
                    "Predictions are estimates and do not certify that water is safe."
                )
