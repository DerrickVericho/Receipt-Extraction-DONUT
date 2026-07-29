import io
import time

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from service import fetch_models, extract_receipt

st.set_page_config(layout="wide", page_title="Donut KIE \u2014 Receipt Extraction")


# CSS style for the app
st.markdown(
    """
<style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Small uppercase section labels, like the wireframe's "1 \u2014 UPLOAD IMAGE" */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.75;
        margin-bottom: 0.6rem;
    }

    /* Title row with badge, like "Donut KIE \u2014 Receipt Extraction  [Research Demo]" */
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-title-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.25);
        opacity: 0.85;
    }

    /* Small muted caption under each model radio option */
    .model-caption {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-top: -0.5rem;
        margin-bottom: 0.4rem;
    }

</style>
""",
    unsafe_allow_html=True,
)


# header
with st.container():
    st.markdown(
        '<div class="app-title">Donut KIE \u2014 Receipt Extraction '
        '<span class="app-title-badge">Research Demo</span></div>',
        unsafe_allow_html=True,
    )

st.write("")


# main layout
col_left, col_right = st.columns([1, 2])


# left side (upload image and select model)
with col_left:

    # upload image
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">1 \u2014 Upload Image</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop receipt image here (.jpg / .png / .webp)",
            type=["jpg", "png", "webp"],
            label_visibility="collapsed",
        )

    st.write("")

    # select model
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">2 \u2014 Select Model</div>',
            unsafe_allow_html=True,
        )

        # model fetching
        model_options = fetch_models()
        if not model_options:
            st.error("app is currently unavailable. Please try again later")
            st.stop()

        selected_model = st.radio(
            "Choose a model",
            list(model_options.keys()),
            format_func=lambda x: model_options[x],
            label_visibility="collapsed",
        )

        st.write("")
        run_inference = st.button(
            "Run Inference", type="primary", use_container_width=True
        )


# right side (inference metrics and result)
with col_right:

    # image preview + interactive crop
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">Image Preview</div>', unsafe_allow_html=True
        )

        cropped_img = None
        if uploaded_file is not None:
            img = Image.open(io.BytesIO(uploaded_file.getvalue()))

            enable_crop = st.checkbox("Crop image (optional)", key="enable_crop")
            if enable_crop:
                cropped_img = st_cropper(
                    img, realtime_update=True, box_color="#FF4B4B", aspect_ratio=None
                )
                st.image(cropped_img, caption="Cropped preview", width=300)
            else:
                st.image(uploaded_file, use_container_width=True)
        else:
            st.info("No image uploaded yet")

    st.write("")

    # inference metrics
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">Inference Metrics</div>', unsafe_allow_html=True
        )
        metric_col1, metric_col2 = st.columns(2)

    st.write("")

    # extracted fields
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">Extracted Fields</div>', unsafe_allow_html=True
        )
        extracted_fields_placeholder = st.empty()

    # inference logic
    if run_inference:

        # upload file validation
        if uploaded_file is not None:
            with st.spinner("Running inference..."):
                start_time = time.time()

                # prepare bytes (cropped or original)
                if cropped_img is not None:
                    buf = io.BytesIO()
                    cropped_img.save(buf, format="PNG")
                    file_bytes = buf.getvalue()
                else:
                    file_bytes = uploaded_file.getvalue()

                # extract receipt
                try:
                    response = extract_receipt(
                        uploaded_file, selected_model, file_bytes
                    )
                    latency = time.time() - start_time

                    # metrics
                    with metric_col1:
                        st.metric(label="Latency", value=f"{latency:.2f}s")
                    with metric_col2:
                        st.metric(
                            label="Status",
                            value=(
                                "Success" if response.status_code == 200 else "Failed"
                            ),
                        )

                    # display extracted fields
                    if response.status_code == 200:
                        res_json = response.json()
                        if res_json.get("success"):
                            extracted_fields_placeholder.json(res_json.get("data", {}))
                        else:
                            extracted_fields_placeholder.error(
                                f"Inference failed: {res_json.get('msg')}"
                            )
                    else:
                        extracted_fields_placeholder.error(
                            f"Backend returned status code {response.status_code}"
                        )

                # if extract receipt error
                except Exception as e:
                    with metric_col1:
                        st.metric(label="Latency", value="\u2013 ms")
                    with metric_col2:
                        st.metric(label="Status", value="Error")
                    extracted_fields_placeholder.error(
                        f"Error connecting to backend: {e}"
                    )

        # if user hasnt uploaded an image
        else:
            with metric_col1:
                st.metric(label="Latency", value="\u2013 ms")
            with metric_col2:
                st.metric(label="Model size", value="\u2013 MB")
            extracted_fields_placeholder.warning("Please upload an image first.")

    # if user has not run inference
    else:
        with metric_col1:
            st.metric(label="Latency", value="\u2013 ms")
        with metric_col2:
            st.metric(label="Model size", value="\u2013 MB")

        extracted_fields_placeholder.info("Run inference to see extracted fields.")
