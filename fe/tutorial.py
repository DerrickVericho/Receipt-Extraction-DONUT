import base64
import mimetypes
import os
import streamlit as st

TUTORIAL_SLIDES = [
    {
        "title": "1. Upload Your Receipt",
        "desc": "Got a receipt? Upload or Drop a photo into the upload zone.",
        "bullets": [
            "**Supported Formats:** JPG, JPEG, PNG, and WEBP.",
            "**Image Quality:** Make sure the text is readable. Blurry photos produce inaccurate extractions.",
            "**Preview & Verification:** You'll see a preview on the left, with 'Clear' / 'Crop' buttons right below it.",
        ],
        "image": "assets/Receipt-Tut.png",
    },
    {
        "title": "2. Crop & Rotate",
        "desc": "Crop your receipt so only the items and the total amount are visible.",
        "bullets": [
            "**Interactive Crop Tool:** Click **Crop** to drag the box around the edges of the receipt.",
            "**Rotation Controls:** Use **Rotate ↺ / ↻** to quickly adjust 90-degree rotations if the photo was taken sideways.",
            "**Confirm or Discard:** Click **Done** to update the working image, or **Cancel** to revert.",
        ],
        "image": "assets/ReceiptCropped-Tut.png",
    },
    {
        "title": "3. Select a Model & Extract",
        "desc": "Pick a model to run the extraction.",
        "bullets": [
            "**Model Dropdown:** Pick a model from the dropdown.",
            "**Trade-offs:** Lighter models save more storage but may be slightly less accurate."
        ],
        "image": "assets/ModelSelection-Tut.png",
    },
    {
        "title": "4. Inspect Results & Export JSON",
        "desc": "Check the extracted items, then grab the results as JSON when you're done.",
        "bullets": [
            "**Formatted Line Items:** Clear readable view separating item names, quantities, unit prices, tax, and totals.",
            "**Expandable Raw JSON:** Click **View Raw JSON** to peek at exactly what the model returned.",
            "**Download JSON:** Click **Download JSON** to instantly save `extraction_result.json` straight to your computer.",
        ],
        "image": "assets/Result-Tut.png",
    },
]


@st.dialog("DONUT Receipt Extraction", width="large")
def show_tutorial_dialog():
    step = st.session_state.get("tutorial_step", 0)
    total_steps = len(TUTORIAL_SLIDES)
    slide = TUTORIAL_SLIDES[step]

    step_cols = st.columns(total_steps)
    for i in range(total_steps):
        with step_cols[i]:
            if st.button(
                f"Step {i + 1}",
                key=f"tut_step_btn_{i}",
                type="primary" if i == step else "secondary",
                use_container_width=True,
            ):
                st.session_state.tutorial_step = i
                st.rerun(scope="fragment")

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    st.progress((step + 1) / total_steps)

    img_val = slide.get("image", "")
    img_src = img_val
    local_path = None
    if os.path.exists(img_val):
        local_path = img_val
    else:
        cand = os.path.join(os.path.dirname(__file__), img_val)
        if os.path.exists(cand):
            local_path = cand

    if local_path:
        with open(local_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        mime_type = mimetypes.guess_type(local_path)[0] or "image/png"
        img_src = f"data:{mime_type};base64,{b64_data}"

    st.markdown(
        f'<div class="tutorial-img-container">'
        f'<img src="{img_src}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;" />'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"### {slide['title']}")
    st.write(slide["desc"])

    for bullet in slide["bullets"]:
        st.markdown(f"- {bullet}")

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    c_prev, c_ind, c_next = st.columns([0.3, 0.4, 0.3])

    with c_prev:
        if st.button("Previous", use_container_width=True, disabled=(step == 0)):
            st.session_state.tutorial_step = max(0, step - 1)
            st.rerun(scope="fragment")

    with c_ind:
        st.markdown(
            f'<div class="step-indicator">Slide {step + 1} of {total_steps}</div>',
            unsafe_allow_html=True,
        )

    with c_next:
        if step < total_steps - 1:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.tutorial_step = min(total_steps - 1, step + 1)
                st.rerun(scope="fragment")
        else:
            if st.button("Done", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 0
                st.rerun()
