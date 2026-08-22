import base64
import mimetypes
import os
import streamlit as st

# tutorial slides config
TUTORIAL_SLIDES = [
    {
        "title": "1. Upload Your Receipt",
        "desc": "Upload a receipt photo from your device using the upload zone or file uploader. The app supports standard photo formats up to 10 MB.",
        "bullets": [
            "**Supported Formats:** JPG, JPEG, PNG, and WEBP.",
            "**Image Quality:** Ensure receipt text is sharp, legible, and unblurred.",
            "**Preview & Verification:** Once uploaded, a preview appears on the left pane with filename details and 'Clear' / 'Crop' buttons.",
        ],
        "image": "assets/Receipt.jpg",
    },
    {
        "title": "2. Crop & Rotate",
        "desc": "Crop your receipt so only the items and the total amount are visible.",
        "bullets": [
            "**Interactive Crop Tool:** Click **Crop** to drag the box around the edges of the receipt.",
            "**Rotation Controls:** Use **Rotate ↺ / ↻** to quickly adjust 90-degree rotations if the photo was taken sideways.",
            "**Confirm or Discard:** Click **Apply crop** to update the working image, or **Cancel** to revert.",
        ],
        "image": "assets/ReceiptCropped.png",
    },
    {
        "title": "3. Select Model & Run Extraction",
        "desc": "Choose which model you want to use. We included different model tiers so you can compare their extraction performance:",
        "bullets": [
            "**DONUT-Base:** The best and most accurate model (recommended).",
            "**DONUT-P50-KD:** A balanced model — faster with solid accuracy.",
            "**DONUT-P70-KD:** A lighter model, slightly lower accuracy.",
            "**DONUT-P30-KD-Q:** The most compressed model — we included this so you can try out how the lowest-tier version performs.",
            "**Execute Extraction:** Pick any model from the dropdown and click **Execute extraction** to start.",
        ],
        "image": "assets/ModelSelection.png",
    },
    {
        "title": "4. Inspect Results & Export JSON",
        "desc": "Review structured line items, quantities, prices, subtotals, and export results for downstream accounting or analysis.",
        "bullets": [
            "**Formatted Line Items:** Clear readable view separating item names, quantities, unit prices, tax, and totals.",
            "**Expandable Raw JSON:** Click **View Raw JSON** to inspect the full nested hierarchy and raw response schema.",
            "**Download JSON:** Click **Download JSON** to instantly save `extraction_result.json` to your local device.",
        ],
        "image": "assets/Result.png",
    },
]


@st.dialog("📖 How to Use Donut KIE", width="large")
def show_tutorial_dialog():
    """Carousel modal tutorial dialog."""
    step = st.session_state.get("tutorial_step", 0)
    total_steps = len(TUTORIAL_SLIDES)
    slide = TUTORIAL_SLIDES[step]

    # step selector pill bar
    step_cols = st.columns(total_steps)
    for i in range(total_steps):
        with step_cols[i]:
            btn_label = f"{'🟢' if i == step else '⚪'} Step {i + 1}"
            if st.button(btn_label, key=f"tut_step_btn_{i}", use_container_width=True):
                st.session_state.tutorial_step = i
                st.rerun()

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    st.progress((step + 1) / total_steps)

    # slide visual (the pictures for each step)
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

    # slide content
    st.markdown(f"### {slide['title']}")
    st.write(slide["desc"])

    for bullet in slide["bullets"]:
        st.markdown(f"- {bullet}")

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    # carousel navigation controls
    c_prev, c_ind, c_next = st.columns([0.3, 0.4, 0.3])

    with c_prev:
        if st.button("⬅️ Previous", use_container_width=True, disabled=(step == 0)):
            st.session_state.tutorial_step = max(0, step - 1)
            st.rerun()

    with c_ind:
        st.markdown(
            f'<div class="step-indicator">Slide {step + 1} of {total_steps}</div>',
            unsafe_allow_html=True,
        )

    with c_next:
        if step < total_steps - 1:
            if st.button("Next ➡️", type="primary", use_container_width=True):
                st.session_state.tutorial_step = min(total_steps - 1, step + 1)
                st.rerun()
        else:
            if st.button("🎉 Got it!", type="primary", use_container_width=True):
                st.session_state.show_tutorial = False
                st.session_state.tutorial_step = 0
                st.rerun()
