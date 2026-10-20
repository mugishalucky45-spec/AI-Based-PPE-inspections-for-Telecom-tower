
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import cv2

st.set_page_config(
    page_title="AI based Tower PPE Inspection",
    page_icon="🦺",
    layout="wide"
)

MODEL_PATH = Path(
    r"C:\ppe_yolov8_clean\ppe_yolov8_clean"
    r"\runs\harness_experiments\img800_trial\weights\best.pt"
)

PPE_CLASSES = [
    "Helmet", "Gloves", "Safety Vest",
    "Safety Shoes", "Harness"
]

if "inspection" not in st.session_state:
    st.session_state.inspection = None

if "worker_status" not in st.session_state:
    st.session_state.worker_status = {}

st.markdown("""
<style>
.stApp { background: #f5f8fb; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.hero {
    background: #16324f; color: white; padding: 25px;
    border-radius: 16px; margin-bottom: 24px;
}
.panel {
    background: white; padding: 22px; border-radius: 16px;
    border: 1px solid #e4eaf0; margin-bottom: 18px;
}
.section-title { color: #16324f; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Telecom Tower PPE Monitor</h1>
  <p>Site inspections • Safety hazards • PPE detection</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SITE SETUP ----------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("## Site Setup")
st.caption("Enter the worker and tower site details.")

with st.form("site_setup"):
    worker_id = st.text_input("Worker ID", value="W001")
    site_id = st.text_input("Site ID", value="RK00")
    site_name = st.text_input("Site Name", value="ETO KICUKIRO")

    hazard = st.radio(
        "Any safety hazards?",
        ["Yes", "No"],
        horizontal=True
    )

    hazard_description = st.text_area(
        "Describe briefly",
        placeholder="Example: Team found bees on site"
    )

    submitted = st.form_submit_button(
        "Start Inspection",
        type="primary",
        use_container_width=True
    )

if submitted:
    if not worker_id.strip() or not site_id.strip() or not site_name.strip():
        st.error("Please enter Worker ID, Site ID, and Site Name.")
    elif hazard == "Yes" and not hazard_description.strip():
        st.error("Please describe the safety hazard.")
    else:
        st.session_state.inspection = {
            "worker_id": worker_id.strip(),
            "site_id": site_id.strip(),
            "site_name": site_name.strip(),
            "hazard": hazard,
            "hazard_description": hazard_description.strip(),
        }
        st.session_state.worker_status[worker_id.strip()] = {
            "site": site_name.strip(),
            "status": "Inspection started"
        }
        st.success(f"Inspection started for {site_name}.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ACTIVE INSPECTION ----------------
if st.session_state.inspection:
    info = st.session_state.inspection

    st.markdown("## Current Inspection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Site", info["site_name"])
    c2.metric("Site ID", info["site_id"])
    c3.metric("Worker ID", info["worker_id"])

    if info["hazard"] == "Yes":
        st.warning("Safety hazard reported: " + info["hazard_description"])
    else:
        st.success("No hazards reported on the form.")

    st.markdown("## PPE Inspection")
    st.caption(
        "Capture an image from your camera to run your trained YOLOv8 model."
    )

    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    photo = st.camera_input("Take a picture for PPE inspection")

    if photo is not None:
        try:
            if not MODEL_PATH.is_file():
                st.error(f"Model file not found: {MODEL_PATH}")
            else:
                with st.spinner("Loading model and checking PPE..."):
                    model = YOLO(str(MODEL_PATH))

                    import numpy as np
                    file_bytes = np.frombuffer(
                        photo.getvalue(), dtype=np.uint8
                    )
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if frame is None:
                        st.error("Could not read the camera image.")
                    else:
                        result = model.predict(
                            frame,
                            imgsz=800,
                            conf=confidence,
                            verbose=False
                        )[0]

                        annotated = result.plot()
                        annotated = cv2.cvtColor(
                            annotated, cv2.COLOR_BGR2RGB
                        )

                        left, right = st.columns([3, 2])

                        with left:
                            st.image(
                                annotated,
                                caption="PPE detection results",
                                use_container_width=True
                            )

                        detected = {}
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            score = float(box.conf[0])
                            if 0 <= class_id < len(PPE_CLASSES):
                                name = PPE_CLASSES[class_id]
                                detected[name] = max(
                                    detected.get(name, 0), score
                                )

                        rows = []
                        for item in PPE_CLASSES:
                            if item in detected:
                                rows.append({
                                    "PPE Item": item,
                                    "Confidence": f"{detected[item]*100:.1f}%",
                                    "Detection": "Detected"
                                })
                            else:
                                rows.append({
                                    "PPE Item": item,
                                    "Confidence": "—",
                                    "Detection": "Not detected"
                                })

                        with right:
                            st.markdown("### Detection Summary")
                            st.dataframe(
                                pd.DataFrame(rows),
                                hide_index=True,
                                use_container_width=True
                            )

                        if len(detected) == len(PPE_CLASSES):
                            st.info(
                                "All five PPE classes were detected. "
                                "A human inspector must still verify the result."
                            )
                        else:
                            st.warning(
                                "One or more PPE classes were not detected. "
                                "This does not prove the worker is missing PPE."
                            )

                        st.session_state.worker_status[
                            info["worker_id"]
                        ] = {
                            "site": info["site_name"],
                            "status": "Image inspected; human verification required"
                        }

        except Exception as error:
            st.error(f"PPE inspection error: {error}")

st.divider()

# ---------------- WORKER STATUS ----------------
st.markdown("## Check Inspection Status")
st.caption("Enter a Worker ID to look up the status recorded in this app.")

with st.form("check_worker"):
    check_id = st.text_input("Worker ID", value="W001")
    check = st.form_submit_button(
        "Check Status",
        type="primary",
        use_container_width=True
    )

if check:
    record = st.session_state.worker_status.get(check_id.strip())
    if record:
        st.info(f"Site: {record['site']}")
        st.write(f"Status: **{record['status']}**")
    else:
        st.warning(
            "No inspection status found for this Worker ID in this session."
        )

st.caption(
    "Educational prototype. Results are not a certified safety clearance. "
    "False detections and missed PPE are possible."
)
