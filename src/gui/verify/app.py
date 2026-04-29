"""
src/gui/verify/app.py
======================
Streamlit Human-in-the-Loop Verification GUI.

Chạy:
    streamlit run src/gui/verify/app.py -- --run-dir data/teacher_runs_20/run_xxx

Tính năng:
  - Duyệt qua toàn bộ teacher outputs
  - Hiển thị ảnh + bounding boxes
  - Hiển thị reasoning, PoT code, evidence
  - Approve / Reject / Skip
  - Chỉnh sửa final_answer trực tiếp
  - Lọc theo trạng thái (Pending / Approved / Rejected / Skipped)
  - Auto-save trạng thái sau mỗi action
  - Export Gold Standard JSONL
  - Hỗ trợ Multi-image (carousel + bbox per image)
  - Keyboard shortcuts: D (Next), A (Prev)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

# ── Compat shim: streamlit-drawable-canvas 0.9.3 + Streamlit >=1.27 ──
# Thư viện canvas gọi st_image.image_to_url (đã bị xoá). React frontend
# LUÔN ghép origin URL phía trước backgroundImageURL, nên data:… URL bị
# hỏng. Giải pháp: đăng ký ảnh vào Streamlit MediaFileManager để nhận
# relative URL (vd: /media/abc.png) mà frontend resolve đúng cách.
import io as _io
from hashlib import md5 as _md5

import numpy as _np
from PIL import Image as _PILImage

import streamlit_drawable_canvas as _sdc


def _pil_to_media_url(img: _PILImage.Image, key: str) -> str:
    """Đăng ký PIL Image vào Streamlit media file manager, trả về URL."""
    from streamlit.runtime import get_instance

    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    media_mgr = get_instance().media_file_mgr
    url = media_mgr.add(
        img_bytes,
        mimetype="image/png",
        coordinates=f"canvas-bg-{key}",
        file_name=f"canvas_bg_{_md5(img_bytes).hexdigest()[:12]}.png",
    )
    return url


def st_canvas(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: _PILImage.Image | None = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: dict | None = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key=None,
) -> _sdc.CanvasResult:
    """Wrapper quanh streamlit-drawable-canvas, sửa lỗi background image."""
    background_image_url = None
    if background_image is not None:
        background_image = _sdc._resize_img(background_image, height, width)
        background_image_url = _pil_to_media_url(
            background_image, key or "default"
        )
        background_color = ""

    initial_drawing = (
        {"version": "4.4.0"} if initial_drawing is None else initial_drawing
    )
    initial_drawing["background"] = background_color

    component_value = _sdc._component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        realtimeUpdateStreamlit=update_streamlit and (drawing_mode != "polygon"),
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=initial_drawing,
        displayToolbar=display_toolbar,
        displayRadius=point_display_radius,
        key=key,
        default=None,
    )
    if component_value is None:
        return _sdc.CanvasResult

    return _sdc.CanvasResult(
        _np.asarray(_sdc._data_url_to_image(component_value["data"])),
        component_value["raw"],
    )

# ── Ensure project root on sys.path ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Vi-QA/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.verify.store import (
    ALL_STATUSES,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SKIPPED,
    VerifyStore,
)
from src.gui.verify.renderer import render_record_images, load_images, normalize_boxes


# ═══════════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ViQA - Human Verify",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
#  Parse CLI args
# ═══════════════════════════════════════════════════════════════════

def get_run_dir() -> Path:
    """Parse --run-dir từ CLI args (sau dấu --)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=False, default=None)
    # Streamlit passes args after -- to sys.argv
    args, _ = parser.parse_known_args()
    if args.run_dir:
        return Path(args.run_dir)
    # Fallback: scan for latest run
    for search_dir in ["data/teacher_runs_20", "data/teacher_runs"]:
        p = PROJECT_ROOT / search_dir
        if p.exists():
            runs = sorted(p.iterdir())
            if runs:
                return runs[-1]
    st.error("⚠️ Không tìm thấy run dir. Vui lòng truyền `--run-dir` khi chạy.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
#  Initialize Store (cached)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def init_store(run_dir: str) -> VerifyStore:
    return VerifyStore(run_dir, data_root=str(PROJECT_ROOT / "data" / "ViInfographicVQA"))


# ═══════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* General dark theme enhancements */
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
    }
    .status-pending { background: #FFA000; color: #000; }
    .status-approved { background: #4CAF50; color: #FFF; }
    .status-rejected { background: #F44336; color: #FFF; }
    .status-skipped { background: #9E9E9E; color: #FFF; }

    /* Profile badge */
    .profile-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        background: #1976D2;
        color: white;
        font-size: 12px;
        font-weight: 500;
    }

    /* Image container */
    .img-container {
        border: 2px solid #444;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 8px;
    }

    /* Action buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-width: 120px;
    }

    /* Code blocks */
    .stCodeBlock {
        max-height: 300px;
    }

    /* Sidebar stats */
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px solid #333;
    }

    /* Progress bar */
    .progress-container {
        background: #333;
        border-radius: 8px;
        height: 8px;
        margin: 8px 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  BBox ↔ Canvas helpers
# ═══════════════════════════════════════════════════════════════════

CANVAS_DISPLAY_WIDTH = 700

_BBOX_COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFAA00",
    "#FF44FF", "#44FFFF", "#FFFF44", "#AA44FF",
]


def _bboxes_to_initial_drawing(
    norm_boxes: list[dict],
    img_idx: int,
    canvas_w: int,
    canvas_h: int,
) -> dict:
    """
    Convert normalized [0-1000] boxes → fabric.js initial_drawing JSON
    cho 1 ảnh cụ thể (img_idx).
    """
    objects = []
    matching = [b for b in norm_boxes if b["img_idx"] == img_idx]
    for i, box in enumerate(matching):
        c = box["coords"]  # [x1, y1, x2, y2] normalized 0-1000
        x1 = c[0] * canvas_w / 1000
        y1 = c[1] * canvas_h / 1000
        x2 = c[2] * canvas_w / 1000
        y2 = c[3] * canvas_h / 1000
        color = _BBOX_COLORS[i % len(_BBOX_COLORS)]
        objects.append({
            "type": "rect",
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "fill": color + "33",
            "stroke": color,
            "strokeWidth": 3,
        })
    return {"version": "4.4.0", "objects": objects}


def _canvas_to_normalized_bboxes(
    canvas_json: dict | None,
    img_idx: int,
    canvas_w: int,
    canvas_h: int,
) -> list[list[int]]:
    """
    Convert fabric.js canvas JSON → list of 5-element normalized bboxes
    [img_idx, x1, y1, x2, y2] in [0-1000].
    """
    if not canvas_json or "objects" not in canvas_json:
        return []
    boxes: list[list[int]] = []
    for obj in canvas_json["objects"]:
        if obj.get("type") != "rect":
            continue
        sx = obj.get("scaleX", 1)
        sy = obj.get("scaleY", 1)
        left = obj["left"]
        top = obj["top"]
        w = obj["width"] * sx
        h = obj["height"] * sy
        x1 = max(0, min(int(left * 1000 / canvas_w), 1000))
        y1 = max(0, min(int(top * 1000 / canvas_h), 1000))
        x2 = max(0, min(int((left + w) * 1000 / canvas_w), 1000))
        y2 = max(0, min(int((top + h) * 1000 / canvas_h), 1000))
        boxes.append([img_idx, x1, y1, x2, y2])
    return boxes


# ═══════════════════════════════════════════════════════════════════
#  Sidebar: Stats + Filter + Navigation
# ═══════════════════════════════════════════════════════════════════

def render_sidebar(store: VerifyStore):
    with st.sidebar:
        st.markdown("## 🔍 ViQA Human Verify")
        st.markdown(f"📁 `{store.run_dir.name}`")

        # ── Progress Stats ───────────────────────────────────
        st.markdown("### 📊 Tiến độ")
        stats = store.get_stats()
        total = stats["total"]

        reviewed = stats[STATUS_APPROVED] + stats[STATUS_REJECTED] + stats[STATUS_SKIPPED]
        pct = (reviewed / total * 100) if total > 0 else 0

        st.markdown(f"**{reviewed}/{total}** đã duyệt ({pct:.1f}%)")

        # Progress bar
        st.progress(pct / 100)

        # Status counts
        cols = st.columns(4)
        labels = [
            ("⏳", STATUS_PENDING),
            ("✅", STATUS_APPROVED),
            ("❌", STATUS_REJECTED),
            ("⏭️", STATUS_SKIPPED),
        ]
        for col, (icon, status) in zip(cols, labels):
            col.metric(icon, stats.get(status, 0))

        st.divider()

        # ── Filter ───────────────────────────────────────────
        st.markdown("### 🎯 Lọc")
        filter_status = st.selectbox(
            "Trạng thái",
            options=["all"] + ALL_STATUSES,
            format_func=lambda x: {
                "all": "📋 Tất cả",
                STATUS_PENDING: "⏳ Chưa duyệt",
                STATUS_APPROVED: "✅ Đã duyệt",
                STATUS_REJECTED: "❌ Từ chối",
                STATUS_SKIPPED: "⏭️ Bỏ qua",
            }.get(x, x),
            key="filter_status",
        )

        filter_multi = st.selectbox(
            "Loại ảnh",
            options=["all", "single", "multi"],
            format_func=lambda x: {
                "all": "📷 Tất cả",
                "single": "🖼️ Single-image",
                "multi": "🖼️🖼️ Multi-image",
            }.get(x, x),
            key="filter_multi",
        )

        # Build filtered list
        if filter_status == "all":
            ids = store.get_ids_by_status(None)
        else:
            ids = store.get_ids_by_status(filter_status)

        # Filter by image count
        if filter_multi == "single":
            ids = [sid for sid in ids
                   if store.get_record(sid) and store.get_record(sid).get("num_images", 1) == 1]
        elif filter_multi == "multi":
            ids = [sid for sid in ids
                   if store.get_record(sid) and store.get_record(sid).get("num_images", 1) > 1]

        st.markdown(f"**{len(ids)}** samples")

        st.divider()

        # ── Export ───────────────────────────────────────────
        st.markdown("### 📤 Export")
        if st.button("🏆 Export Gold Standard", use_container_width=True):
            count = store.export_gold()
            st.success(f"✅ Exported {count} records → `gold_standard.jsonl`")

        # ── Reviewer Name ────────────────────────────────────
        st.divider()
        reviewer = st.text_input("👤 Reviewer", value="reviewer_1", key="reviewer_name")

    return ids, reviewer


# ═══════════════════════════════════════════════════════════════════
#  Main Content: Sample Viewer
# ═══════════════════════════════════════════════════════════════════

def render_sample(store: VerifyStore, sample_id: str, reviewer: str, all_ids: list[str]):
    record = store.get_record(sample_id)
    if not record:
        st.error(f"Sample {sample_id} not found!")
        return

    state = store.get_state(sample_id)
    status = state["status"]

    # ── Header ───────────────────────────────────────────────────
    header_cols = st.columns([3, 1, 1, 1])
    with header_cols[0]:
        st.markdown(f"### Sample `{sample_id}`")
    with header_cols[1]:
        badge_class = f"status-{status}"
        st.markdown(
            f'<span class="status-badge {badge_class}">{status.upper()}</span>',
            unsafe_allow_html=True,
        )
    with header_cols[2]:
        profile = record.get("output_profile", "unknown")
        st.markdown(
            f'<span class="profile-badge">{profile}</span>',
            unsafe_allow_html=True,
        )
    with header_cols[3]:
        num_imgs = record.get("num_images", 1)
        st.markdown(f"🖼️ **{num_imgs} ảnh**")

    st.divider()

    # ── Layout: Images (left) | Details (right) ──────────────────
    img_col, detail_col = st.columns([1, 1])

    # ── Left: Images with BBox ───────────────────────────────────
    canvas_results: dict[int, tuple] = {}  # img_idx → (canvas_result, cw, ch)

    with img_col:
        st.markdown("#### 📸 Ảnh + BBox")

        # Toggle edit mode
        edit_bbox = st.checkbox(
            "✏️ Chỉnh sửa BBox",
            value=False,
            key=f"edit_bbox_{sample_id}",
        )

        # Load raw images (without bbox overlay)
        try:
            raw_images = load_images(record, str(PROJECT_ROOT))
        except Exception as e:
            st.error(f"Lỗi load ảnh: {e}")
            raw_images = []

        norm_boxes = normalize_boxes(
            record.get("grounding_boxes", []),
            num_images=len(raw_images),
        )

        if not edit_bbox:
            # ── Static view (ảnh đã vẽ BBox sẵn, read-only) ──────
            try:
                rendered_images = render_record_images(record, str(PROJECT_ROOT))
            except Exception:
                rendered_images = raw_images

            if len(rendered_images) == 1:
                st.image(rendered_images[0], use_container_width=True)
            elif len(rendered_images) > 1:
                img_tabs = st.tabs([f"Ảnh {i}" for i in range(len(rendered_images))])
                for i, tab in enumerate(img_tabs):
                    with tab:
                        p = record.get("image_paths", [])
                        st.caption(f"📂 `{p[i]}`" if i < len(p) else "")
                        st.image(rendered_images[i], use_container_width=True)
        else:
            # ── Editable canvas (interactive BBox drawing) ────────
            mode_col, color_col = st.columns([2, 1])
            with mode_col:
                drawing_mode = st.radio(
                    "Chế độ",
                    options=["transform", "rect"],
                    format_func=lambda x: {
                        "transform": "🔄 Di chuyển / Resize",
                        "rect": "➕ Vẽ BBox mới",
                    }.get(x, x),
                    horizontal=True,
                    key=f"draw_mode_{sample_id}",
                )
            with color_col:
                stroke_color = st.color_picker(
                    "Màu", value="#FF4444",
                    key=f"bbox_color_{sample_id}",
                )

            def _render_canvas(img, idx):
                w, h = img.size
                cw = CANVAS_DISPLAY_WIDTH
                ch = int(cw * h / w)
                initial = _bboxes_to_initial_drawing(norm_boxes, idx, cw, ch)
                result = st_canvas(
                    fill_color=stroke_color + "33",
                    stroke_width=3,
                    stroke_color=stroke_color,
                    background_image=img,
                    drawing_mode=drawing_mode,
                    initial_drawing=initial,
                    width=cw,
                    height=ch,
                    key=f"canvas_{sample_id}_{idx}",
                )
                canvas_results[idx] = (result, cw, ch)

            if len(raw_images) == 1:
                _render_canvas(raw_images[0], 0)
            elif len(raw_images) > 1:
                img_tabs = st.tabs([f"Ảnh {i}" for i in range(len(raw_images))])
                for i, tab in enumerate(img_tabs):
                    with tab:
                        p = record.get("image_paths", [])
                        st.caption(f"📂 `{p[i]}`" if i < len(p) else "")
                        _render_canvas(raw_images[i], i)

        # Raw bbox info (reference)
        raw_boxes = record.get("grounding_boxes", [])
        if raw_boxes:
            with st.expander(f"📐 Bounding Boxes gốc ({len(raw_boxes)})", expanded=False):
                for i, box in enumerate(raw_boxes):
                    if len(box) == 5:
                        st.code(f"Box {i}: Ảnh {box[0]} → [{box[1]}, {box[2]}, {box[3]}, {box[4]}]")
                    else:
                        st.code(f"Box {i}: [{', '.join(str(x) for x in box)}]")

    # ── Right: Question/Answer/Reasoning ─────────────────────────
    with detail_col:
        # Question
        st.markdown("#### ❓ Câu hỏi")
        st.info(record.get("question", "—") if "question" in record else
                record.get("run_meta", {}).get("question", "—"))

        # Ground truth (if available)
        gt_answer = record.get("answer")
        if gt_answer:
            st.markdown("#### 🎯 Ground Truth")
            st.success(gt_answer)

        # Final Answer
        st.markdown("#### 💡 Final Answer (Teacher)")
        st.warning(record.get("final_answer", "—"))

        # Reasoning
        reasoning = record.get("reasoning")
        if reasoning:
            with st.expander("🧠 Reasoning", expanded=False):
                st.markdown(reasoning)

        # Python code (PoT)
        python_code = record.get("python_code")
        if python_code:
            with st.expander("🐍 Python Code (PoT)", expanded=False):
                st.code(python_code, language="python")

        # Evidence
        evidence_text = record.get("evidence_text")
        if evidence_text:
            with st.expander("📋 Evidence Text", expanded=False):
                st.markdown(evidence_text)

        evidence_items = record.get("evidence_items", [])
        if evidence_items:
            with st.expander(f"📋 Evidence Items ({len(evidence_items)})", expanded=False):
                for item in evidence_items:
                    st.json(item)

        # Validation
        validation = record.get("validation", {})
        if validation:
            with st.expander("✔️ Validation", expanded=False):
                for k, v in validation.items():
                    icon = "✅" if v else "❌"
                    st.markdown(f"{icon} `{k}`")

    st.divider()

    # ── Action Panel ─────────────────────────────────────────────
    st.markdown("#### 🎬 Hành động")

    # Editable answer
    edited_answer = st.text_area(
        "✏️ Chỉnh sửa Answer (để trống = giữ nguyên)",
        value=state.get("edited_answer") or "",
        height=80,
        key=f"edit_answer_{sample_id}",
    )

    # Note
    note = st.text_input(
        "📝 Ghi chú",
        value=state.get("note") or "",
        key=f"note_{sample_id}",
    )

    # ── Collect edited boxes from canvas (nếu đang ở edit mode) ──
    edited_boxes = None
    if canvas_results:
        all_edited: list[list[int]] = []
        for idx, (cr, cw, ch) in canvas_results.items():
            if cr and cr.json_data:
                all_edited.extend(_canvas_to_normalized_bboxes(cr.json_data, idx, cw, ch))
        edited_boxes = all_edited or None

    # Action buttons
    btn_cols = st.columns([1, 1, 1, 1, 1, 1])

    current_idx = all_ids.index(sample_id) if sample_id in all_ids else 0

    with btn_cols[0]:
        if st.button("✅ Approve", key=f"approve_{sample_id}",
                      type="primary", use_container_width=True):
            store.update_status(
                sample_id, STATUS_APPROVED,
                edited_answer=edited_answer.strip() or None,
                edited_boxes=edited_boxes,
                note=note.strip() or None,
                reviewer=reviewer,
            )
            # Auto next
            if current_idx + 1 < len(all_ids):
                st.session_state["current_idx"] = current_idx + 1
            st.rerun()

    with btn_cols[1]:
        if st.button("❌ Reject", key=f"reject_{sample_id}",
                      use_container_width=True):
            store.update_status(
                sample_id, STATUS_REJECTED,
                edited_answer=edited_answer.strip() or None,
                edited_boxes=edited_boxes,
                note=note.strip() or None,
                reviewer=reviewer,
            )
            if current_idx + 1 < len(all_ids):
                st.session_state["current_idx"] = current_idx + 1
            st.rerun()

    with btn_cols[2]:
        if st.button("⏭️ Skip", key=f"skip_{sample_id}",
                      use_container_width=True):
            store.update_status(
                sample_id, STATUS_SKIPPED,
                edited_boxes=edited_boxes,
                note=note.strip() or None,
                reviewer=reviewer,
            )
            if current_idx + 1 < len(all_ids):
                st.session_state["current_idx"] = current_idx + 1
            st.rerun()

    with btn_cols[3]:
        if st.button("🔄 Reset", key=f"reset_{sample_id}",
                      use_container_width=True):
            store.update_status(sample_id, STATUS_PENDING, reviewer=reviewer)
            st.rerun()

    with btn_cols[4]:
        if st.button("◀️ Prev (A)", key=f"prev_{sample_id}",
                      use_container_width=True,
                      disabled=(current_idx == 0)):
            st.session_state["current_idx"] = max(0, current_idx - 1)
            st.rerun()

    with btn_cols[5]:
        if st.button("▶️ Next (D)", key=f"next_{sample_id}",
                      use_container_width=True,
                      disabled=(current_idx >= len(all_ids) - 1)):
            st.session_state["current_idx"] = min(len(all_ids) - 1, current_idx + 1)
            st.rerun()

    # Review history
    if state.get("reviewed_at"):
        st.caption(
            f"🕐 Last reviewed: {state['reviewed_at']} by {state.get('reviewer', '?')}"
        )


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    inject_css()

    run_dir = get_run_dir()
    store = init_store(str(run_dir))

    # Sidebar
    filtered_ids, reviewer = render_sidebar(store)

    if not filtered_ids:
        st.info("🎉 Không có sample nào trong bộ lọc này!")
        return

    # Navigation state
    if "current_idx" not in st.session_state:
        st.session_state["current_idx"] = 0

    # Clamp index
    idx = st.session_state["current_idx"]
    idx = max(0, min(idx, len(filtered_ids) - 1))
    st.session_state["current_idx"] = idx

    # Sample selector
    with st.sidebar:
        st.divider()
        st.markdown("### 🔢 Chuyển nhanh")
        jump_idx = st.number_input(
            f"Index (0 → {len(filtered_ids) - 1})",
            min_value=0,
            max_value=max(0, len(filtered_ids) - 1),
            value=idx,
            step=1,
            key="jump_idx",
        )
        if jump_idx != idx:
            st.session_state["current_idx"] = jump_idx
            st.rerun()

    # Render current sample
    current_id = filtered_ids[idx]
    render_sample(store, current_id, reviewer, filtered_ids)


if __name__ == "__main__":
    main()
