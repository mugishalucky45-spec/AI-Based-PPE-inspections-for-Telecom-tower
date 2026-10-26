import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from ultralytics import YOLO
from datetime import datetime, timezone

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(os.environ.get(
    "PPE_MODEL_PATH",
    str(APP_DIR / "best.pt")
))

BUCKET = "inspection-photos"
PPE_CLASSES = ["helmet", "gloves", "safety_vest", "safety_shoes", "harness"]
REGION_RANGES = {
    "Kigali": ("K", 1, 400),
    "Eastern": ("E", 1, 400),
    "Western": ("W", 1, 400),
    "Southern": ("S", 1, 400),
    "Northern": ("N", 1, 30),
}
STATUS_LABELS = {
    "UNDER REVIEW": "UNDER REVIEW",
    "APPROVED": "APPROVED",
    "REJECTED": "REJECTED",
}

st.set_page_config(page_title="TowerSafe | PPE Inspection", page_icon="🦺", layout="wide")


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def analyze_image(image_bytes: bytes, confidence: float = 0.25):
    if not MODEL_PATH.exists():
        return None, (
            f"YOLO model file not found: {MODEL_PATH}. "
            f"App folder files: {[p.name for p in APP_DIR.iterdir()]}"
        )

    try:
        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if frame is None:
            return None, "Could not read the camera image."

      
        model = load_model(str(MODEL_PATH))
       

        result = model.predict(
            frame,
            imgsz=800,
            conf=0.70,
            verbose=False,
        )[0]

        detections = []

        names = result.names

        # Temporary debugging information
        st.session_state["yolo_class_names"] = names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = str(names.get(class_id, class_id))

                detections.append({
                    "class": class_name,
                    "confidence": round(float(box.conf[0].item()), 3),
                    "box": [
                        round(float(v), 1)
                        for v in box.xyxy[0].tolist()
                    ],
                })

        counts = {name: 0 for name in PPE_CLASSES}

        for detection in detections:
            class_name = detection["class"].strip().lower()

            if class_name in counts:
                counts[class_name] += 1

        ppe_status = {}

        for ppe_class in PPE_CLASSES:
            ppe_status[ppe_class] = (
                "Detected"
                if counts[ppe_class] > 0
                else "Missing"
            )

        return {
            "detections": detections,
            "counts": counts,
            "ppe_status": ppe_status,
        }, None

    except Exception as exc:
        return None, f"Model prediction failed: {type(exc).__name__}: {exc}"
# Display the saved YOLO class names
if "yolo_class_names" in st.session_state:
    st.write(
        "DEBUG - YOLO model class names:",
        st.session_state["yolo_class_names"]
    )

def make_client() -> Client:
    # Read Streamlit secrets when configured; environment variables are a local fallback.
    try:
        url = str(st.secrets["SUPABASE_URL"]).strip()
        anon_key = str(st.secrets["SUPABASE_ANON_KEY"]).strip()
    except Exception:
        url = os.environ.get("SUPABASE_URL", "").strip()
        anon_key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not anon_key:
        st.error("Supabase connection is not configured yet. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit secrets.")
        st.code('SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"\nSUPABASE_ANON_KEY = "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY"', language="toml")
        st.stop()
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(url, anon_key)
    return st.session_state.supabase_client

def get_user_id(user):
    if not user:
        return None
    return getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)

def get_user_email(user):
    if not user:
        return ""
    return getattr(user, "email", "") or (user.get("email", "") if isinstance(user, dict) else "")


def get_user_role(client, user_id):
    """Return the user's role from the user_roles table."""
    try:
        response = (
            client.table("user_roles")
            .select("role")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]["role"]
    except Exception as e:
        st.error(f"Could not load user role: {e}")

    return "worker"

def get_inspector_role(client, user_id):
    try:
        response = client.table("inspector_roles").select("inspector_name").eq("user_id", user_id).limit(1).execute()
        rows = response.data or []
        return rows[0].get("inspector_name") if rows else None
    except Exception:
        return None

def fetch_inspections(client, user_id, is_inspector):
    query = client.table("inspections").select("*").order("submitted_at", desc=True)
    if not is_inspector:
        query = query.eq("worker_user_id", user_id)
    response = query.execute()
    return response.data or []

def fetch_notifications(client):
    response = client.table("notifications").select("*").order("created_at", desc=True).limit(30).execute()
    return response.data or []

def upload_photo(client, user_id, inspection_code, label, image_bytes):
    # User-specific folder matches the Storage INSERT policy configured in Supabase.
    path = f"{user_id}/{inspection_code}/{label}.jpg"
    client.storage.from_(BUCKET).upload(
        path,
        image_bytes,
        {"content-type": "image/jpeg", "upsert": "true"},
    )
    return path

def show_private_photo(client, path, caption):
    if not path:
        st.warning(f"No {caption.lower()} was stored for this inspection.")
        return
    try:
        image_bytes = client.storage.from_(BUCKET).download(path)
        st.image(image_bytes, caption=caption, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not retrieve {caption.lower()}. Check the private Storage SELECT policies. Details: {exc}")

def status_message(status):
    if status == "APPROVED":
        st.success("APPROVED — The inspector has approved this inspection. Follow your site's safety procedures before continuing.")
    elif status == "REJECTED":
        st.error("REJECTED — Do not continue yet. Correct the PPE issues and follow your supervisor's instructions.")
    else:
        st.warning("UNDER REVIEW — Your photos are waiting for the inspector. Do not continue until approval is recorded.")

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
.stApp label, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3 {color:#172b43;}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {background:#ffffff !important; color:#172b43 !important; border-color:#718096 !important;}
.stRadio label, .stCheckbox label {color:#172b43 !important;}
.stButton button, .stFormSubmitButton button {background:#075985 !important; color:#ffffff !important; border:1px solid #075985 !important;}
.stAlert, [data-testid="stAlert"] {color:#172b43 !important;}
</style>
""", unsafe_allow_html=True)

client = make_client()

with st.sidebar:
    st.markdown("## 🦺 TowerSafe")
    st.caption("Shared online PPE inspection")
    st.divider()

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# Reuse an existing authenticated user from Streamlit session state.
user = st.session_state.auth_user
if user is None:
    st.markdown('<div class="hero"><h1>TowerSafe PPE Inspection</h1><p>Sign in to submit or review PPE inspections.</p></div>', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            login_clicked = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if login_clicked:
            try:
                result = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                st.session_state.auth_user = result.user
                st.session_state.auth_email = email.strip()
                st.rerun()
            except Exception as exc:
                st.error(f"Sign-in failed: {exc}")
    with signup_tab:
        st.info("Create an account for a worker first. Inspector access must be assigned by the project administrator in Supabase; creating an account alone does not grant inspector permissions.")
        with st.form("signup_form"):
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input("Password (at least 8 characters recommended)", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_clicked = st.form_submit_button("Create worker account", type="primary", use_container_width=True)
        if signup_clicked:
            if not new_email.strip() or len(new_password) < 8:
                st.error("Enter an email address and a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    result = client.auth.sign_up({"email": new_email.strip(), "password": new_password})
                    if getattr(result, "session", None):
                        st.session_state.auth_user = result.user
                        st.session_state.auth_email = new_email.strip()
                        st.success("Account created. You are signed in.")
                        st.rerun()
                    else:
                        st.success("Account created. Check your email for the confirmation link, then return here and sign in.")
                except Exception as exc:
                    st.error(f"Account creation failed: {exc}")
    st.stop()

user_id = get_user_id(user)
email = get_user_email(user) or st.session_state.get("auth_email", "")
if not user_id:
    st.error("Could not identify the signed-in account. Please sign out and sign in again.")
    if st.button("Return to sign in"):
        st.session_state.auth_user = None
        st.rerun()
    st.stop()

inspector_name_from_role = get_inspector_role(client, user_id)
user_role = get_user_role(client, user_id)
is_inspector = user_role in ("inspector", "administrator")
is_administrator = user_role == "administrator"
with st.sidebar:
    st.write(f"Signed in: {email or user_id}")
    st.write(f"Role: {'Inspector' if is_inspector else 'Worker'}")
    if st.button("Sign out", use_container_width=True):
        try:
            client.auth.sign_out()
        except Exception:
            pass
        st.session_state.auth_user = None
        st.session_state.pop("active_inspection", None)
        st.rerun()
   
if is_administrator:
    page = st.radio(
        "Administrator menu",
        ["Inspector review", "Inspection records", "Notifications", "Tower site management"]
    )
elif is_inspector:
    page = st.radio(
        "Inspector menu",
        ["Inspector review", "Inspection records", "Notifications", "Tower site management"]
    )
else:
    page = st.radio(
        "Worker menu",
        ["Worker inspection", "Worker status", "Notifications"]
    )


if page == "Tower site management":
    st.title("Tower Site Management")
    st.write("Register and manage MTN and Airtel tower sites.")

    with st.form("register_tower_site"):
        site_id = st.text_input("Site ID")
        site_name = st.text_input("Site Name")
        province = st.selectbox(
            "Province",
            ["Kigali City", "Eastern Province", "Western Province",
             "Southern Province", "Northern Province"]
        )
        district = st.text_input("District")
        sector = st.text_input("Sector")
        activity = st.selectbox("Activity", ["MTN", "Airtel"])

        submitted = st.form_submit_button("Register Tower Site")

    if submitted:
        if not all([site_id.strip(), site_name.strip(),
                    district.strip(), sector.strip()]):
            st.error("Please complete all tower site fields.")
        else:
            try:
                client.table("tower_sites").insert({
                    "site_id": site_id.strip(),
                    "site_name": site_name.strip(),
                    "province": province,
                    "district": district.strip(),
                    "sector": sector.strip(),
                    "activity": activity,
                }).execute()
                st.success("Tower site registered successfully.")
            except Exception as e:
                st.error(
                    "Could not register the tower site. "
                    "Check whether the Site ID already exists and verify database permissions."
                )

    st.subheader("Registered Tower Sites")
    try:
        sites = (
            client.table("tower_sites")
            .select("site_id, site_name, province, district, sector, activity")
            .order("site_id")
            .execute()
        )
        st.dataframe(sites.data, use_container_width=True)
    except Exception:
        st.error("Could not load tower sites. Check database permissions.")

    st.stop()



st.markdown('<div class="hero"><h1>TowerSafe PPE Inspection</h1><p>Front and back photo then inspector review.</p></div>', unsafe_allow_html=True)
st.caption("PLEASE INSERT BELOW INFORMATION.")

try:
    items = fetch_inspections(client, user_id, is_inspector)
except Exception as exc:
    st.error(f"Could not load inspection records from Supabase: {exc}")
    st.stop()

pending_count = sum(1 for item in items if item.get("status") == "UNDER REVIEW")
if is_inspector:
    st.sidebar.metric("Inspections waiting", pending_count)

if page == "Worker inspection":
    st.subheader("1. Worker and site details")

    # Load registered tower sites from Supabase
    try:
        response = (
            client.table("tower_sites")
            .select("site_id, site_name, province, district, sector, activity")
            .order("site_id")
            .execute()
        )
        sites = response.data or []
    except Exception as exc:
        st.error(f"Could not load registered tower sites: {exc}")
        st.stop()

    if not sites:
        st.warning("No tower sites are registered yet.")
        st.stop()

    site_options = {
        f"{site['site_id']} — {site['site_name']} ({site['activity']})": site
        for site in sites
    }

    # Show the details form only when no inspection is active.
    if not st.session_state.get("active_inspection"):
        with st.form("worker_details"):
            technician = st.text_input(
                "Technician name",
                placeholder="Enter full name",
                key="technician_input",
            )

            selected_label = st.selectbox(
                "Select tower site",
                list(site_options.keys()),
                key="selected_tower_site",
            )

            start_clicked = st.form_submit_button(
                "Start inspection",
                type="primary",
                use_container_width=True,
            )

        if start_clicked:
            selected_site = site_options[selected_label]

            if not technician.strip():
                st.error("Please enter the technician name.")
            else:
                st.session_state.active_inspection = {
                    "inspection_code": (
                        "TS-"
                        + datetime.now().strftime("%Y%m%d-%H%M%S")
                        + "-"
                        + uuid.uuid4().hex[:5].upper()
                    ),
                    "technician": technician.strip(),
                    "region": selected_site["province"],
                    "site_id": selected_site["site_id"],
                    "site_name": selected_site["site_name"],
                }

                # Clear previous photos only when starting a new inspection.
                for key in [
                    "front_capture",
                    "back_capture",
                    "left_capture",
                    "right_capture",
                ]:
                    st.session_state.pop(key, None)

                for key in [
                    "front_camera",
                    "back_camera",
                    "left_camera",
                    "right_camera",
                ]:
                    st.session_state.pop(key, None)

                st.rerun()

    # Keep the active inspection visible on subsequent reruns.
    active = st.session_state.get("active_inspection")

    if active:
        st.divider()
        st.subheader("2. Capture live camera evidence")

        st.write(
            f"**Technician:** {active['technician']} · "
            f"**Site:** {active['site_id']} — {active['site_name']} · "
            f"**Region:** {active['region']}"
        )

        st.caption(
            "Allow camera access when your browser asks. "
            "Capture all four views before submitting."
        )

        # Front and Back
        front_col, back_col = st.columns(2)

        with front_col:
            st.markdown("#### Front view")
            front = st.camera_input(
                "Take front-view photo",
                key="front_camera",
            )
            if front is not None:
                st.session_state.front_capture = front.getvalue()

        with back_col:
            st.markdown("#### Back view")
            back = st.camera_input(
                "Take back-view photo",
                key="back_camera",
            )
            if back is not None:
                st.session_state.back_capture = back.getvalue()

        # Left and Right
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### Left-side view")
            left = st.camera_input(
                "Take left-side photo",
                key="left_camera",
            )
            if left is not None:
                st.session_state.left_capture = left.getvalue()

        with right_col:
            st.markdown("#### Right-side view")
            right = st.camera_input(
                "Take right-side photo",
                key="right_camera",
            )
            if right is not None:
                st.session_state.right_capture = right.getvalue()

        # Submit all four photos
        if st.button(
            "Submit evidence to inspector",
            type="primary",
            use_container_width=True,
        ):
            front_bytes = st.session_state.get("front_capture")
            back_bytes = st.session_state.get("back_capture")
            left_bytes = st.session_state.get("left_capture")
            right_bytes = st.session_state.get("right_capture")

            if not all([
                front_bytes,
                back_bytes,
                left_bytes,
                right_bytes,
            ]):
                st.error(
                    "Please capture all four views: "
                    "front, back, left, and right."
                )
            else:
                try:
                    inspection_code = active["inspection_code"]

                    # Analyze all four photos
                    front_analysis, front_error = analyze_image(front_bytes)
                    back_analysis, back_error = analyze_image(back_bytes)
                    left_analysis, left_error = analyze_image(left_bytes)
                    right_analysis, right_error = analyze_image(right_bytes)

                    # Upload all four photos
                    front_path = upload_photo(
                        client, user_id, inspection_code, "front", front_bytes
                    )
                    back_path = upload_photo(
                        client, user_id, inspection_code, "back", back_bytes
                    )
                    left_path = upload_photo(
                        client, user_id, inspection_code, "left", left_bytes
                    )
                    right_path = upload_photo(
                        client, user_id, inspection_code, "right", right_bytes
                    )

                    # Store analysis results
                    ppe_results = {
                        "inspection_code": inspection_code,
                        "region": active["region"],
                        "front_analysis": front_analysis,
                        "back_analysis": back_analysis,
                        "left_analysis": left_analysis,
                        "right_analysis": right_analysis,
                        "left_photo_path": left_path,
                        "right_photo_path": right_path,
                        "analysis_errors": [
                            err for err in [
                                front_error,
                                back_error,
                                left_error,
                                right_error,
                            ] if err
                        ],
                    }

                    # Prepare the database record
                    insert_data = {
                        "technician_name": active["technician"],
                        "site_id": active["site_id"],
                        "site_name": active["site_name"],
                        "worker_user_id": user_id,
                        "status": "UNDER REVIEW",
                        "front_photo_path": front_path,
                        "back_photo_path": back_path,
                        "ppe_results": ppe_results,
                    }

                    # Insert inspection into Supabase
                    client.table("inspections").insert(
                        insert_data
                    ).execute()

                    st.success(
                        f"Evidence submitted. Reference: {inspection_code}"
                    )
                    st.warning(
                        "UNDER REVIEW — Do not continue working until "
                        "the inspector records approval."
                    )

                    # Clear the active inspection only after successful submission.
                    for key in [
                        "active_inspection",
                        "front_capture",
                        "back_capture",
                        "left_capture",
                        "right_capture",
                        "front_camera",
                        "back_camera",
                        "left_camera",
                        "right_camera",
                        "technician_input",
                        "selected_tower_site",
                    ]:
                        st.session_state.pop(key, None)

                    st.rerun()

                except Exception as exc:
                    st.error(f"Submission failed: {exc}")
                    st.info(
                        "If a photo uploaded before the error, it may remain "
                        "in private Storage. We can clean up incomplete uploads "
                        "after verifying the setup."
                    )

elif page == "Worker status":
    st.subheader("Your inspection status")
    st.caption("This list comes from the shared Supabase database and is limited to your signed-in account.")
    if st.button("Refresh status"):
       st.rerun()
    if not items:
        st.info("You have not submitted an inspection yet.")
    for rec in items:
        ppe = rec.get("ppe_results") or {}
        code = ppe.get("inspection_code") or str(rec.get("id", ""))
        with st.container(border=True):
            st.markdown(f"**{rec.get('site_id')} — {rec.get('site_name')}**")
            st.caption(f"Reference: {code} · Submitted: {rec.get('submitted_at', 'Unknown time')}")
            status = rec.get("status", "UNDER REVIEW")
            status_message(status)
            st.write(f"**Inspector:** {rec.get('inspector_name') or 'Not reviewed yet'}")
            st.write(f"**Comments:** {rec.get('inspector_comments') or 'No comments yet'}")

elif page == "Inspector review":
    if not is_inspector:
        st.error("This account has not been authorized as an inspector.")
        st.stop()
    # Device Monitoring Site selector
    st.divider()
    st.subheader("Device Monitoring Site")

    try:
        sites_response = (
            client.table("tower_sites")
            .select("site_id, site_name")
            .order("site_id")
            .execute()
        )
        device_sites = sites_response.data or []

        if not device_sites:
            st.warning("No tower sites are available.")
        else:
            site_options = {
                f"{site['site_id']} — {site['site_name']}": site
                for site in device_sites
            }

            current_setting = (
                client.table("device_monitoring_settings")
                .select("site_id, site_name")
                .eq("id", 1)
                .execute()
            )

            saved_site = (
                current_setting.data[0]
                if current_setting.data
                else None
            )

            labels = list(site_options.keys())
            default_index = 0

            if saved_site:
                for index, label in enumerate(labels):
                    if site_options[label]["site_id"] == saved_site["site_id"]:
                        default_index = index
                        break

            # --- ESP8266: Multiple Device Monitoring Sites ---
st.subheader("ESP8266 Device Monitoring Sites")

# Load available tower sites.
sites_response = client.table("tower_sites").select(
    "site_id, site_name"
).execute()

sites = sites_response.data or []

if not sites:
    st.warning("No tower sites are available.")
else:
    # Create labels for the multiselect.
    site_options = {
        f"{site['site_id']} — {site['site_name']}": site
        for site in sites
    }

    labels = list(site_options.keys())

    # Load currently selected monitoring sites.
    saved_response = (
        client.table("device_monitoring_sites")
        .select("site_id, site_name")
        .eq("selected", True)
        .execute()
    )

    saved_sites = saved_response.data or []
    saved_ids = {site["site_id"] for site in saved_sites}

    default_labels = [
        label
        for label, site in site_options.items()
        if site["site_id"] in saved_ids
    ]

    selected_labels = st.multiselect(
        "Choose all tower sites for the ESP8266",
        options=labels,
        default=default_labels,
        key="device_monitoring_sites",
    )

    if st.button("Save Device Monitoring Sites"):
        selected_sites = [
            site_options[label]
            for label in selected_labels
        ]

        # Mark every existing monitoring site as unselected.
        (
            client.table("device_monitoring_sites")
            .update({
                "selected": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .neq("site_id", "")
            .execute()
        )

        # Save the newly selected sites.
        if selected_sites:
            rows = [
                {
                    "site_id": site["site_id"],
                    "site_name": site["site_name"],
                    "selected": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                for site in selected_sites
            ]

            (
                client.table("device_monitoring_sites")
                .upsert(rows, on_conflict="site_id")
                .execute()
            )

        st.success(
            f"Saved {len(selected_sites)} monitoring site(s)."
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Could not load or save the monitoring site: {exc}")
    pending = [rec for rec in items if rec.get("status") == "UNDER REVIEW"]
    st.subheader(f"Pending inspections ({len(pending)})")
    if st.button("Refresh pending inspections"):
       st.rerun()
    if not pending:
        st.success("No inspections are waiting for review.")
    else:
        options = {}
        for rec in pending:
            ppe = rec.get("ppe_results") or {}
            code = ppe.get("inspection_code") or str(rec.get("id", ""))
            options[f"{code} · {rec.get('technician_name')} · {rec.get('site_id')}"] = rec
        selected_label = st.selectbox("Choose an inspection", list(options.keys()))
        rec = options[selected_label]
        ppe = rec.get("ppe_results") or {}
        st.write(f"**Technician:** {rec.get('technician_name')} | **Region:** {ppe.get('region', 'Not recorded')} | **Site:** {rec.get('site_id')} — {rec.get('site_name')}")
        col_front, col_back = st.columns(2)
        def make_ppe_status_table(analysis):
            display_names = {
                "helmet": "Helmet",
                "gloves": "Safety Gloves",
                "safety_vest": "Reflective Vest",
                "safety_shoes": "Safety Boots",
                "harness": "Full-Body Safety Harness",
            }

            if not analysis or "counts" not in analysis:
                return None

            counts = analysis["counts"]

            rows = []
            for ppe_class in PPE_CLASSES:
                detected = counts.get(ppe_class, 0) > 0

                rows.append({
                    "PPE Item": display_names.get(
                        ppe_class, ppe_class.replace("_", " ").title()
                    ),
                    "Status": "Detected" if detected else "Missing",
                })

            return pd.DataFrame(rows)

        with col_front:
            st.markdown("#### Front view")
            show_private_photo(
                client, rec.get("front_photo_path"), "Front-view evidence"
            )

            front_analysis = ppe.get("front_analysis")
            front_table = make_ppe_status_table(front_analysis)

            if front_table is not None:
                st.dataframe(
                    front_table,
                    hide_index=True,
                    use_container_width=True,
                )
            elif front_analysis:
                st.warning("PPE analysis is unavailable. Inspect the photo manually.")
            else:
                st.info("Model analysis unavailable; inspect the photo manually.")

        with col_back:
            st.markdown("#### Back view")
            show_private_photo(
                client, rec.get("back_photo_path"), "Back-view evidence"
            )

            back_analysis = ppe.get("back_analysis")
            back_table = make_ppe_status_table(back_analysis)

            if back_table is not None:
                st.dataframe(
                    back_table,
                    hide_index=True,
                    use_container_width=True,
                )
            elif back_analysis:
                st.warning("PPE analysis is unavailable. Inspect the photo manually.")
            else:
                st.info("Model analysis unavailable; inspect the photo manually.")
        for err in ppe.get("analysis_errors", []):
            st.warning(err)
        st.warning("YOLOv8 checks helmet, gloves, safety vest, safety shoes, and harness. It can miss or misclassify PPE. The inspector makes the final decision.")
        with st.form("review_form"):
            reviewer = st.text_input("Inspector name", value=inspector_name_from_role or "")
            decision = st.radio("Decision", ["APPROVED — worker may continue", "REJECTED — worker must correct PPE"], horizontal=True)
            comments = st.text_area("Review comments / required corrections")
            submit_review = st.form_submit_button("Save inspector decision", type="primary")
        if submit_review:
            if not reviewer.strip():
                st.error("Enter the inspector name.")
            elif decision.startswith("REJECTED") and not comments.strip():
                st.error("Please describe what the worker must correct before resubmitting.")
            else:
                try:
                    new_status = "APPROVED" if decision.startswith("APPROVED") else "REJECTED"
                    client.table("inspections").update({
                        "status": new_status,
                        "inspector_name": reviewer.strip(),
                        "inspector_comments": comments.strip(),
                        "decided_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", rec["id"]).execute()
                    st.success("Decision saved to the shared database.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save the decision: {exc}")

elif page == "Inspection records":
    if not is_inspector:
        st.error("This account has not been authorized as an inspector.")
        st.stop()
    st.subheader("All inspection records")
    if not items:
        st.info("No inspection records have been submitted yet.")
    else:
        df = pd.DataFrame([{
            "Reference": (rec.get("ppe_results") or {}).get("inspection_code", rec.get("id")),
            "Submitted": rec.get("submitted_at"),
            "Technician": rec.get("technician_name"),
            "Site ID": rec.get("site_id"),
            "Site name": rec.get("site_name"),
            "Status": rec.get("status"),
            "Inspector": rec.get("inspector_name") or "—",
            "Comments": rec.get("inspector_comments") or "—",
        } for rec in items])
        status_filter = st.multiselect("Filter by status", sorted(df["Status"].dropna().unique()), default=sorted(df["Status"].dropna().unique()))
        st.dataframe(df[df["Status"].isin(status_filter)], hide_index=True, use_container_width=True)
        st.download_button("Download records as CSV", df[df["Status"].isin(status_filter)].to_csv(index=False).encode("utf-8"), file_name="towersafe_inspections.csv", mime="text/csv")

elif page == "Notifications":
    st.subheader("Notifications")
    st.caption("Notifications are read from the shared database. Database triggers must be installed to create notifications when inspections are submitted or decided.")
    if st.button("Refresh notifications"):
        st.rerun()
    try:
        notifications = fetch_notifications(client)
        if not notifications:
            st.info("No notifications yet.")
        for note in notifications:
            with st.container(border=True):
                st.markdown(f"**{note.get('title', 'Notification')}**")
                st.write(note.get("message", ""))
                st.caption(note.get("created_at", ""))
    except Exception as exc:
        st.error(f"Could not load notifications: {exc}")
