import base64
import io
import json
import time

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from service import fetch_models, extract_receipt

st.set_page_config(layout="wide", page_title="Donut KIE \u2014 Receipt Extraction")

st.markdown(
    """
<style>
    .block-container { max-width: 1400px; padding-top: 1rem; }
    .img-frame {
        width: 100%;
        height: 340px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        background: rgba(255,255,255,0.02);
    }
    .img-frame img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        cursor: pointer;
    }
    .upload-zone {
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 2.5rem 1rem;
        text-align: center;
        cursor: pointer;
    }
    .meta-row {
        font-size: 0.85rem;
        opacity: 0.75;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

DEFAULTS = {
    "original_bytes": None,
    "original_name": None,
    "original_type": None,
    "cropped_bytes": None,
    "cropping": False,
    "rotate_angle": 0,
    "selected_model": None,
    "result": None,
    "error": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.upload_key += 1
    st.rerun()


def get_image_bytes():
    return st.session_state.cropped_bytes or st.session_state.original_bytes


# ── Title ──

st.markdown("## Donut KIE \u2014 Receipt Extraction")

col_left, col_right = st.columns([0.35, 0.65])

# ══════════════════════════════════════════════════════════════
# LEFT COLUMN
# ══════════════════════════════════════════════════════════════

with col_left:

    # ── Crop mode ──
    if st.session_state.cropping:
        raw_bytes = get_image_bytes()
        if raw_bytes is not None:
            img = Image.open(io.BytesIO(raw_bytes))
            angle = st.session_state.rotate_angle
            if angle != 0:
                img = img.rotate(angle, expand=True)

            cropped = st_cropper(
                img, realtime_update=True, box_color="#FF4B4B", aspect_ratio=None
            )

            c_apply, c_cancel = st.columns(2)
            with c_apply:
                if st.button("Apply crop", use_container_width=True):
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    st.session_state.cropped_bytes = buf.getvalue()
                    st.session_state.cropping = False
                    st.session_state.rotate_angle = 0
                    st.rerun()
            with c_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.cropping = False
                    st.session_state.rotate_angle = 0
                    st.rerun()

            st.markdown("&nbsp;")
            c_rl, c_rr, c_rs = st.columns(3)
            with c_rl:
                if st.button("Rotate \u21ba", use_container_width=True):
                    st.session_state.rotate_angle = (
                        st.session_state.rotate_angle + 90
                    ) % 360
                    st.rerun()
            with c_rr:
                if st.button("\u21bb Rotate", use_container_width=True):
                    st.session_state.rotate_angle = (
                        st.session_state.rotate_angle - 90
                    ) % 360
                    st.rerun()
            with c_rs:
                if st.button("Reset", use_container_width=True):
                    st.session_state.rotate_angle = 0
                    st.rerun()

    # ── Normal mode ──
    else:
        if st.session_state.original_bytes is None:
            with st.container():
                st.markdown(
                    '<div class="upload-zone"><b>Upload receipt image</b><br>'
                    '<span style="font-size:0.8rem;opacity:0.6;">'
                    "JPG, JPEG, PNG, WEBP  &middot;  Max 10 MB</span></div>",
                    unsafe_allow_html=True,
                )
                uploaded = st.file_uploader(
                    "Upload",
                    type=["jpg", "jpeg", "png", "webp"],
                    label_visibility="collapsed",
                    key=f"upload_{st.session_state.upload_key}",
                )
                if uploaded is not None:
                    if uploaded.size > 10 * 1024 * 1024:
                        st.error("File too large. Maximum 10 MB.")
                    else:
                        st.session_state.original_bytes = uploaded.getvalue()
                        st.session_state.original_name = uploaded.name
                        st.session_state.original_type = uploaded.type
                        st.session_state.cropped_bytes = None
                        st.session_state.result = None
                        st.session_state.error = None
                        st.rerun()
        else:
            with st.container():
                st.markdown(
                    f'<div style="font-size:0.75rem;opacity:0.55;margin-bottom:0.2rem;">'
                    f"{st.session_state.original_name}</div>",
                    unsafe_allow_html=True,
                )
                img = Image.open(io.BytesIO(get_image_bytes()))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(
                    f'<div class="img-frame">'
                    f'<img src="data:image/png;base64,{img_b64}" />'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            c_clr, c_crp = st.columns(2)
            with c_clr:
                if st.button("Clear", use_container_width=True):
                    reset_all()
            with c_crp:
                if st.button("Crop", use_container_width=True):
                    st.session_state.cropping = True
                    st.session_state.rotate_angle = 0
                    st.rerun()

    st.divider()

    # ── Model selector ──
    models = fetch_models()
    if not models:
        st.error("No models available from backend.")
        st.stop()

    model_map = {m["id"]: m for m in models}
    model_opts = list(model_map.keys())
    default_idx = 0
    for i, m in enumerate(models):
        if m.get("recommended"):
            default_idx = i
            break

    selected_id = st.selectbox(
        "Select model",
        model_opts,
        index=default_idx,
        format_func=lambda rid: model_map[rid]["name"],
    )
    st.session_state.selected_model = selected_id

    # ── Execute button ──
    has_image = st.session_state.original_bytes is not None
    run = st.button(
        "Execute extraction",
        type="primary",
        use_container_width=True,
        disabled=not has_image,
    )

# ══════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════════════════════

with col_right:
    st.markdown("### Extraction output")

    # trigger extraction when Execute is clicked
    if run and has_image:
        with st.spinner("Extracting document..."):
            file_bytes = get_image_bytes()
            start = time.time()
            model_name = model_map[selected_id]["name"]
            try:
                resp = extract_receipt(
                    selected_id,
                    file_bytes,
                    st.session_state.original_name,
                    st.session_state.original_type,
                )
                latency = time.time() - start
                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("success"):
                        st.session_state.result = {
                            "data": body.get("data", {}),
                            "latency": latency,
                            "model_name": model_name,
                        }
                        st.session_state.error = None
                    else:
                        st.session_state.error = body.get(
                            "msg", "Extraction failed"
                        )
                        st.session_state.result = None
                else:
                    st.session_state.error = (
                        f"Backend error: {resp.status_code}"
                    )
                    st.session_state.result = None
            except Exception as e:
                st.session_state.error = f"Connection error: {e}"
                st.session_state.result = None
            st.rerun()

    result = st.session_state.result
    error = st.session_state.error

    if result:
        prediction = result.get("data", {}).get("prediction", {})
        meta_name = result.get("model_name", "")
        meta_lat = result.get("latency", 0)

        st.markdown(
            f'<div class="meta-row">{meta_name} &middot; {meta_lat:.2f}s &middot; Success</div>',
            unsafe_allow_html=True,
        )

        raw = json.dumps(prediction, indent=2, ensure_ascii=False)
        raw_escaped = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div style="max-height:500px;overflow-y:auto;border:1px solid rgba(255,255,255,0.1);'
            f'border-radius:8px;padding:0.75rem;background:rgba(255,255,255,0.02);">'
            f'<pre style="margin:0;white-space:pre-wrap;font-size:0.78rem;font-family:monospace;">'
            f'<code>{raw_escaped}</code></pre></div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "Download JSON",
            raw,
            file_name="extraction_result.json",
            mime="application/json",
            use_container_width=True,
        )

    elif error:
        st.error("Extraction failed")
        st.write(error)
        if st.button("Try again", use_container_width=True):
            st.session_state.error = None
            st.rerun()

    else:
        st.info(
            "Extraction result will appear here.\n\n"
            "Upload a receipt, select a model, and click Execute."
        )
