import base64
import io
import json
import threading
import time

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from service import fetch_models, extract_receipt, preload_model
from tutorial import show_tutorial_dialog

st.set_page_config(layout = "wide", page_title = "DONUT Receipt Extraction")

st.markdown(
    """
<style>
    .block-container { max-width: 1400px; padding-top: 2rem; }
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
    /* Tutorial Modal & Carousel Styles */
    .tutorial-img-container {
        width: 100%;
        height: 480px;
        background: radial-gradient(circle at center, rgba(99, 102, 241, 0.15) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .step-indicator {
        text-align: center;
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.7;
        margin: 0.4rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# session state
CROP_DISPLAY_MAX_W = 360
CROP_DISPLAY_MAX_H = 480

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
    "crop_rect": None,
    "result": None,
    "error": None,
    "tutorial_step": 0,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "prefetched_models" not in st.session_state:
    st.session_state.prefetched_models = []


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.upload_key += 1
    st.rerun()


def get_image_bytes():
    return st.session_state.cropped_bytes or st.session_state.original_bytes


def _fit_display(img, max_w, max_h):
    ratio = min(max_w / img.width, max_h / img.height, 1.0)
    return img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))


def _rotate(delta):
    st.session_state.rotate_angle = (st.session_state.rotate_angle + delta) % 360
    st.rerun()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_models_cached():
    models = fetch_models()
    if models is None:
        raise ConnectionError("Model catalog unavailable")
    return models


def format_simple_receipt(data):
    if not isinstance(data, dict):
        return str(data)

    lines = []

    # menu
    menu = data.get("menu") or data.get("items") or []
    if isinstance(menu, dict):
        menu = [menu]
    elif not isinstance(menu, list):
        menu = []

    for item in menu:
        if isinstance(item, dict):
            cnt = str(item.get("cnt") or item.get("num") or "1").replace("x", "").replace("X", "").strip()
            nm = str(item.get("nm") or item.get("name") or "Unknown item").strip()
            unitprice = str(item.get("unitprice") or item.get("unit_price") or "").strip()
            price = str(item.get("price") or item.get("total_price") or unitprice).strip()

            if unitprice and price and unitprice != price:
                lines.append(f"{cnt} {nm} {unitprice}, Total: {price}")
            elif price:
                lines.append(f"{cnt} {nm} {price}, Total: {price}")
            else:
                lines.append(f"{cnt} {nm}")
        else:
            lines.append(str(item))

    # subtotal
    subtotal = data.get("sub_total") or data.get("subtotal") or {}
    if isinstance(subtotal, dict):
        for k, v in subtotal.items():
            k_label = k.replace("_", " ").title()
            lines.append(f"{k_label}: {v}")

    # total
    total = data.get("total") or data.get("totals") or {}
    if isinstance(total, dict):
        field_names = {
            "total_price": "Total amount",
            "cashprice": "Cash",
            "changeprice": "Change",
            "menuqty_cnt": "Total qty",
            "creditcardprice": "Card",
            "emoneyprice": "E-Money",
            "tax_price": "Tax",
            "discount_price": "Discount",
            "service_price": "Service",
        }
        for k, v in total.items():
            label = field_names.get(k.lower(), k.replace("_", " ").title())
            lines.append(f"{label}: {v}")

    # other fields
    for k, v in data.items():
        if k.lower() not in ["menu", "items", "total", "totals", "sub_total", "subtotal"]:
            if isinstance(v, (dict, list)):
                lines.append(f"{k.replace('_', ' ').title()}: {json.dumps(v, ensure_ascii=False)}")
            else:
                lines.append(f"{k.replace('_', ' ').title()}: {v}")

    return "\n".join(lines) if lines else str(data)


# header (title and tutorial button)

col_header_left, col_header_right = st.columns([0.8, 0.2], vertical_alignment="center")
with col_header_left:
    st.markdown("## DONUT Receipt Extraction")
with col_header_right:
    if st.button("Guide", use_container_width=True):
        st.session_state.tutorial_step = 0
        show_tutorial_dialog()

col_left, col_right = st.columns([0.35, 0.65])

# left column

with col_left:

    # crop mode
    if st.session_state.cropping:
        raw_bytes = get_image_bytes()
        if raw_bytes is not None:
            src = Image.open(io.BytesIO(raw_bytes))
            angle = st.session_state.rotate_angle
            if angle != 0:
                src = src.rotate(angle, expand=True)

            disp = _fit_display(src.copy(), CROP_DISPLAY_MAX_W, CROP_DISPLAY_MAX_H)
            box = st_cropper(
                disp,
                realtime_update=True,
                box_color="#FF4B4B",
                aspect_ratio=None,
                return_type="box",
                should_resize_image=False,
            )
            st.session_state.crop_rect = box

            c_apply, c_cancel = st.columns(2)
            with c_apply:
                if st.button("Done", use_container_width=True):
                    rect = st.session_state.crop_rect or {}
                    sx = src.width / max(1, disp.width)
                    sy = src.height / max(1, disp.height)
                    left = max(0, int(rect.get("left", 0) * sx))
                    top = max(0, int(rect.get("top", 0) * sy))
                    right = min(src.width, int((rect.get("left", 0) + rect.get("width", disp.width)) * sx))
                    bottom = min(src.height, int((rect.get("top", 0) + rect.get("height", disp.height)) * sy))
                    cropped = src.crop((left, top, right, bottom))
                    buf = io.BytesIO()
                    cropped.convert("RGB").save(buf, format="JPEG", quality=90)
                    st.session_state.cropped_bytes = buf.getvalue()
                    st.session_state.cropping = False
                    st.session_state.rotate_angle = 0
                    st.session_state.crop_rect = None
                    st.rerun()
            with c_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.cropping = False
                    st.session_state.rotate_angle = 0
                    st.session_state.crop_rect = None
                    st.rerun()

            st.markdown("&nbsp;")
            c_rl, c_rr, c_rs = st.columns(3)
            with c_rl:
                if st.button("Rotate \u21ba", use_container_width=True):
                    _rotate(90)
            with c_rr:
                if st.button("\u21bb Rotate", use_container_width=True):
                    _rotate(-90)
            with c_rs:
                if st.button("Reset", use_container_width=True):
                    st.session_state.rotate_angle = 0
                    st.rerun()

    # normal mode
    else:
        if st.session_state.original_bytes is None:
            with st.container():
                st.markdown(
                    '<div class="upload-zone"><b>Drop a receipt here, or click to upload</b><br>'
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
                        st.error("That file's too large. Max size is 10 MB.")
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
                is_cropped = st.session_state.cropped_bytes is not None
                preview_bytes = (
                    st.session_state.cropped_bytes if is_cropped
                    else st.session_state.original_bytes
                )
                mime = "image/jpeg" if is_cropped else (st.session_state.original_type or "image/png")
                img_b64 = base64.b64encode(preview_bytes).decode()
                st.markdown(
                    f'<div class="img-frame">'
                    f'<img src="data:{mime};base64,{img_b64}" />'
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

    # model selector
    try:
        models = _fetch_models_cached()
    except ConnectionError:
        models = None
    if not models:
        st.error("No models are available right now.")
        st.stop()

    model_map = {m["id"]: m for m in models}
    model_opts = list(model_map.keys())
    default_idx = next((i for i, m in enumerate(models) if m.get("recommended")), 0)

    selected_id = st.selectbox(
        "Select model",
        model_opts,
        index=default_idx,
        format_func=lambda rid: model_map[rid]["name"],
    )
    st.session_state.selected_model = selected_id

    if selected_id not in st.session_state.prefetched_models:
        st.session_state.prefetched_models.append(selected_id)
        if not model_map[selected_id].get("loaded"):
            threading.Thread(
                target=preload_model,
                args=(selected_id,),
                daemon=True,
            ).start()
            st.toast(f"Getting {model_map[selected_id]['name']} ready…")

    has_image = st.session_state.original_bytes is not None
    run = st.button(
        "Extract",
        type="primary",
        use_container_width=True,
        disabled=not has_image,
    )

# right column

with col_right:
    st.markdown("### Results")

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
                    st.session_state.result = {
                        "data": resp.json().get("data", {}),
                        "latency": latency,
                        "model_name": model_name,
                    }
                    st.session_state.error = None
                else:
                    # backend kirim pesan error di "detail" (HTTPException)
                    st.session_state.error = resp.json().get(
                        "detail", f"Backend error: {resp.status_code}"
                    )
                    st.session_state.result = None
            except Exception:
                st.session_state.error = "Couldn't reach the server. Please try again."
                st.session_state.result = None
            st.rerun()

    result = st.session_state.result
    error = st.session_state.error

    if result:
        prediction = result.get("data", {}).get("prediction", {})
        timing = result.get("data", {}).get("timing") or {}
        meta_name = result.get("model_name", "")
        meta_lat = timing.get("inference_s") or result.get("latency") or 0
        load_s = timing.get("load_s") or 0
        load_note = (
            f' <span style="opacity:0.55;">&middot; (first-load +{load_s:.2f}s)</span>'
            if load_s
            else ""
        )

        st.markdown(
            f'<div class="meta-row">Done in {meta_lat:.2f}s {load_note}</div>',
            unsafe_allow_html=True,
        )

        simple_text = format_simple_receipt(prediction)
        raw = json.dumps(prediction, indent=2, ensure_ascii=False)

        # Simple human-readable output
        st.code(simple_text, language=None)

        with st.expander("View Raw JSON", expanded=False):
            st.code(raw, language="json")

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
            "Upload a receipt to get started."
        )
