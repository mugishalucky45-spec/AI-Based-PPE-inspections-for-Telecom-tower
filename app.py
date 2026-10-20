import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
INSPECTIONS_FILE = DATA_DIR / "inspections.json"
MODEL_PATH = Path(os.environ.get(
    "PPE_MODEL_PATH",
    r"C:\ppe_yolov8_clean\ppe_yolov8_clean\runs\harness_experiments\img800_trial\weights\best.pt"
))
PPE_CLASSES = ["helmet", "gloves", "safety_vest", "safety_shoes", "harness"]
REGION_RANGES = {
    "Kigali": ("K", 1, 400),
    "Eastern": ("E", 1, 400),
    "Western": ("W", 1, 400),
    "Southern": ("S", 1, 400),
    "Northern": ("N", 1, 30),
}

st.set_page_config(page_title="TowerSafe | PPE Inspection", page_icon="🦺", layout="wide")

def read_json(path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return default

def write_json(path, data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def site_id_error(region, site_id):
    prefix, minimum, maximum = REGION_RANGES[region]
    value = (site_id or "").strip().upper()
    match = re.fullmatch(r"([A-Z])(\d{3})", value)
    if not match:
        return f"Invalid site ID. For {region}, enter {prefix}{minimum:03d} to {prefix}{maximum:03d}."
    entered_prefix, digits = match.group(1), int(match.group(2))
    if entered_prefix != prefix or not minimum <= digits <= maximum:
        return f"Invalid site ID for {region}. Please enter {prefix}{minimum:03d} to {prefix}{maximum:03d}."
    return None

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

def analyze_image(image_bytes, confidence):
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            return None, "Could not read the camera image."
        model = load_model(str(MODEL_PATH))
        result = model.predict(frame, imgsz=800, conf=confidence, verbose=False)[0]
        detections = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                detections.append({
                    "class": str(names.get(cls_id, cls_id)),
                    "confidence": round(float(box.conf[0].item()), 3),
                })
        counts = {name: 0 for name in PPE_CLASSES}
        for d in detections:
            if d["class"] in counts:
                counts[d["class"]] += 1
        return {"detections": detections, "counts": counts}, None
    except Exception as exc:
        return None, f"Model prediction failed: {exc}"

def save_capture(inspection_id, label, uploaded_file):
    folder = EVIDENCE_DIR / inspection_id
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg"
    target = folder / f"{label}{suffix}"
    target.write_bytes(uploaded_file.getvalue())
    return str(target.relative_to(APP_DIR))

def get_items():
    return read_json(INSPECTIONS_FILE, [])

def update_items(items):
    write_json(INSPECTIONS_FILE, items)

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {background:#f4f7fb !important; color:#172b43 !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {background:#ffffff !important;}
[data-testid="stSidebar"] * {color:#172b43 !important;}
.block-container {padding-top:1.5rem; max-width:1400px;}
.hero {padding:1.35rem 1.5rem; border-radius:18px; color:#ffffff !important; background:linear-gradient(115deg,#102b50,#176b87);}
.hero h1, .hero p {color:#ffffff !important;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.45rem 0 0 0; opacity:.98;}
.stApp label, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3 {color:#172b43;}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {background:#ffffff !important; color:#172b43 !important; border-color:#718096 !important;}
.stRadio label, .stCheckbox label {color:#172b43 !important;}
.stButton button, .stFormSubmitButton button {background:#075985 !important; color:#ffffff !important; border:1px solid #075985 !important;}
.stAlert, [data-testid="stAlert"] {color:#172b43 !important;}
[data-testid="stDataFrame"] {background:#ffffff !important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🦺 TowerSafe")
    st.caption("Tower worker PPE inspection")
    sidebar_items = read_json(INSPECTIONS_FILE, [])
    pending_count = sum(1 for item in sidebar_items if item.get("status") == "Pending review")
    user_role = st.radio("Open dashboard", ["Worker dashboard", "Inspector dashboard"])
    if user_role == "Worker dashboard":
        page = st.radio("Worker menu", ["Worker inspection", "Worker status"])
    else:
        review_label = f"Inspector review ({pending_count} pending)"
        chosen_inspector_page = st.radio("Inspector menu", [review_label, "Inspection records"])
        page = "Inspector review" if chosen_inspector_page == review_label else chosen_inspector_page
        st.metric("Inspections waiting", pending_count)
    st.divider()
    st.caption("Prototype: local data storage")

st.markdown('<div class="hero"><h1>TowerSafe PPE Inspection</h1><p>Capture front and back evidence, then wait for an inspector decision.</p></div>', unsafe_allow_html=True)
st.write("")

items = get_items()

if page == "Worker inspection":
    worker_identity = st.session_state.get("worker_identity")
    if worker_identity:
        worker_records = [
            item for item in get_items()
            if item.get("technician", "").strip().casefold() == worker_identity["technician"].strip().casefold()
            and item.get("site_id", "").strip().upper() == worker_identity["site_id"].strip().upper()
        ]
        if worker_records:
            latest_worker_record = sorted(worker_records, key=lambda item: item.get("created_at", ""), reverse=True)[0]
            worker_status = latest_worker_record.get("status", "Pending review")
            st.subheader("Your inspection notification")
            if worker_status == "Approved":
                st.success("APPROVED — The inspector has approved your inspection. You may continue at this site. Do not start another inspection unless site procedures require it.")
            elif worker_status == "Rejected":
                st.error("REJECTED — Do not continue working yet. Correct the PPE issues and follow your supervisor's instructions.")
            else:
                st.warning("UNDER REVIEW — Your photos are waiting for the inspector. Do not continue until approval is recorded.")
            st.write(f"**Site:** {latest_worker_record.get('site_id')} — {latest_worker_record.get('site_name')}")
            st.write(f"**Inspector:** {latest_worker_record.get('reviewer') or 'Not reviewed yet'}")
            st.write(f"**Comments:** {latest_worker_record.get('review_comments') or 'No comments yet'}")
            if st.button("Refresh decision", key="refresh_worker_decision"):
                st.rerun()
            st.divider()
    st.subheader("1. Worker and site details")
    st.info("Enter your technician name, site ID and site name. The site ID is validated automatically.")
    with st.form("worker_details"):
        c1, c2 = st.columns(2)
        with c1:
            technician = st.text_input("Technician Name", placeholder="Enter full name")
            site_id = st.text_input("Site ID", placeholder="Example: K001").strip().upper()
        with c2:
            site_name = st.text_input("Site Name", placeholder="Enter tower/site name")
        start_clicked = st.form_submit_button("Start inspection", type="primary", use_container_width=True)

    if start_clicked:
        region = next((r for r, (prefix, low, high) in REGION_RANGES.items()
                       if re.fullmatch(prefix + r"\d{3}", site_id)
                       and low <= int(site_id[1:]) <= high), None)
        error = None
        if not re.fullmatch(r"[A-Z]\d{3}", site_id) or region is None:
            error = "Invalid site ID. Please enter a valid site ID."
        if not technician.strip():
            st.error("Please enter the Technician Name.")
        elif error:
            st.error(error)
        elif not site_name.strip():
            st.error("Please enter the Site Name.")
        else:
            st.session_state["active_inspection"] = {
                "inspection_id": "TS-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5].upper(),
                "technician": technician.strip(),
                "region": region,
                "site_id": site_id,
                "site_name": site_name.strip(),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state.pop("front_capture", None)
            st.session_state.pop("back_capture", None)
            st.success("Details accepted. Capture the front and back views below.")

    active = st.session_state.get("active_inspection")
    if active:
        st.divider()
        st.subheader("2. Capture live camera evidence")
        st.write(f"**Technician:** {active['technician']}  ·  **Site:** {active['site_id']} — {active['site_name']}  ·  **Region:** {active['region']}")
        st.caption("Your browser will ask permission to use the webcam. Streamlit requires a camera interaction by the user; it cannot silently switch on the camera automatically.")
        col_front, col_back = st.columns(2)
        with col_front:
            st.markdown("#### Front view")
            front = st.camera_input("Take front-view photo", key="front_camera")
            if front is not None:
                st.session_state["front_capture"] = front
                st.image(front, caption="Front-view evidence", use_container_width=True)
        with col_back:
            st.markdown("#### Back view")
            back = st.camera_input("Take back-view photo", key="back_camera")
            if back is not None:
                st.session_state["back_capture"] = back
                st.image(back, caption="Back-view evidence", use_container_width=True)

        if st.button("Submit evidence to inspector", type="primary", use_container_width=True):
            front_file = st.session_state.get("front_capture")
            back_file = st.session_state.get("back_capture")
            if front_file is None or back_file is None:
                st.error("Please capture both front and back views before submitting.")
            else:
                existing = [x for x in get_items() if x.get("inspection_id") == active["inspection_id"]]
                if existing:
                    st.warning("This inspection has already been submitted.")
                else:
                    inspection_id = active["inspection_id"]
                    front_path = save_capture(inspection_id, "front", front_file)
                    back_path = save_capture(inspection_id, "back", back_file)
                    confidence = 0.25
                    front_analysis, front_error = analyze_image(front_file.getvalue(), confidence)
                    back_analysis, back_error = analyze_image(back_file.getvalue(), confidence)
                    record = {
                        **active,
                        "status": "Pending review",
                        "front_image": front_path,
                        "back_image": back_path,
                        "front_analysis": front_analysis,
                        "back_analysis": back_analysis,
                        "analysis_errors": [e for e in [front_error, back_error] if e],
                        "reviewer": "",
                        "reviewed_at": "",
                        "review_comments": "",
                    }
                    all_items = get_items()
                    all_items.insert(0, record)
                    update_items(all_items)
                    st.session_state["worker_identity"] = {
                        "technician": active["technician"],
                        "site_id": active["site_id"],
                    }
                    st.session_state.pop("active_inspection", None)
                    st.session_state.pop("front_capture", None)
                    st.session_state.pop("back_capture", None)
                    st.success(f"Evidence submitted. Inspection ID: {inspection_id}.")
                    st.warning("UNDER REVIEW — Your photos have been submitted to the inspector. Do not continue working until the inspector records an approval.")
                    st.info("You can check your decision later in Worker dashboard → Worker status.")
                    st.rerun()



elif page == "Inspector review":
    pending = [x for x in items if x.get("status") == "Pending review"]
    st.subheader(f"Pending inspections ({len(pending)})")
    st.caption("Review each submission below. The worker's status changes when you save a decision.")
    if not pending:
        st.success("No inspections are waiting for review.")
    else:
        options = {f"{x['inspection_id']} · {x['technician']} · {x['site_id']}": x["inspection_id"] for x in pending}
        selected_label = st.selectbox("Choose an inspection", list(options.keys()))
        rec = next(x for x in pending if x["inspection_id"] == options[selected_label])
        st.write(f"**Technician:** {rec['technician']}  |  **Region:** {rec['region']}  |  **Site:** {rec['site_id']} — {rec['site_name']}")
        a, b = st.columns(2)
        for col, key, label in [(a, "front_image", "Front view"), (b, "back_image", "Back view")]:
            with col:
                st.markdown(f"#### {label}")
                image_path = APP_DIR / rec.get(key, "")
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning("Evidence image is missing.")
                analysis_key = "front_analysis" if key == "front_image" else "back_analysis"
                analysis = rec.get(analysis_key)
                if analysis and analysis.get("detections"):
                    st.dataframe(pd.DataFrame(analysis["detections"]), hide_index=True, use_container_width=True)
                elif analysis:
                    st.warning("The model did not detect any of the configured PPE classes in this view.")
                else:
                    st.info("Model analysis unavailable; inspect the image manually.")
        if rec.get("analysis_errors"):
            for err in rec["analysis_errors"]:
                st.warning(err)
        st.warning("YOLOv8 checks only the five trained PPE classes: helmet, gloves, safety vest, safety shoes, and harness. It assists the inspector; it can miss items or detect them incorrectly. The inspector makes the final decision.")
        with st.form("review_form"):
            reviewer = st.text_input("Inspector name")
            decision = st.radio("Decision", ["Approve — worker may continue", "Reject — worker must correct PPE"], horizontal=True)
            comments = st.text_area("Review comments / required corrections")
            submit_review = st.form_submit_button("Save inspector decision", type="primary")
        if submit_review:
            if not reviewer.strip():
                st.error("Enter the inspector name.")
            elif decision.startswith("Reject") and not comments.strip():
                st.error("Please describe what the worker must correct before resubmitting.")
            else:
                all_items = get_items()
                for entry in all_items:
                    if entry.get("inspection_id") == rec["inspection_id"]:
                        entry["status"] = "Approved" if decision.startswith("Approve") else "Rejected"
                        entry["reviewer"] = reviewer.strip()
                        entry["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
                        entry["review_comments"] = comments.strip()
                update_items(all_items)
                st.success("Decision saved.")
                st.rerun()

elif page == "Worker status":
    st.subheader("Check inspection decision")
    st.write("Workers can check whether the inspector approved or rejected their latest inspection.")
    technician_lookup = st.text_input("Technician Name", placeholder="Enter the name used for inspection")
    site_lookup = st.text_input("Site ID", placeholder="Example: K001").strip().upper()
    if st.button("Check status", type="primary"):
        matches = [x for x in get_items()
                   if x.get("technician", "").strip().casefold() == technician_lookup.strip().casefold()
                   and x.get("site_id", "").strip().upper() == site_lookup]
        if not matches:
            st.error("No inspection found for that technician and site ID.")
        else:
            latest = sorted(matches, key=lambda x: x.get("created_at", ""), reverse=True)[0]
            status = latest.get("status", "Pending review")
            if status == "Approved":
                st.success("INSPECTOR DECISION: APPROVED")
                st.markdown("### You are cleared to continue at this site.")
                st.info("Your current inspection is complete. Do not start another inspection unless your supervisor or site procedure requires a new check.")
            elif status == "Rejected":
                st.error("INSPECTOR DECISION: REJECTED")
                st.markdown("### Do not continue work yet.")
                st.write("Correct the PPE issues listed below, then follow your supervisor's instructions for a new inspection.")
            else:
                st.warning("INSPECTOR DECISION: UNDER REVIEW")
                st.markdown("### Your photos are waiting for inspector review.")
                st.write("Do not continue work until the inspector approves your inspection.")
            st.write(f"**Inspection ID:** {latest.get('inspection_id')}")
            st.write(f"**Site:** {latest.get('site_id')} — {latest.get('site_name')}")
            st.write(f"**Submitted:** {latest.get('created_at')}")
            st.write(f"**Inspector:** {latest.get('reviewer') or 'Not reviewed yet'}")
            st.write(f"**Review comments / corrections:** {latest.get('review_comments') or 'No comments yet'}")
            if latest.get("reviewed_at"):
                st.caption(f"Decision recorded: {latest.get('reviewed_at')}")

elif page == "Inspection records":
    st.subheader("All inspection records")
    if not items:
        st.info("No inspection records have been submitted yet.")
    else:
        df = pd.DataFrame([{
            "Inspection ID": x.get("inspection_id"),
            "Date": x.get("created_at"),
            "Technician": x.get("technician"),
            "Region": x.get("region"),
            "Site ID": x.get("site_id"),
            "Site name": x.get("site_name"),
            "Status": x.get("status"),
            "Inspector": x.get("reviewer") or "—",
            "Comments": x.get("review_comments") or "—",
        } for x in items])
        status_filter = st.multiselect("Filter by status", sorted(df["Status"].dropna().unique()), default=sorted(df["Status"].dropna().unique()))
        st.dataframe(df[df["Status"].isin(status_filter)], hide_index=True, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download records as CSV", csv, file_name="towersafe_inspections.csv", mime="text/csv")


