import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np 
import pandas as pd
import streamlit as st

from supabase import create_client
from ultralytics import YOLO


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TowerSafe | PPE Inspection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", str(APP_DIR / "best.pt"))
)

PHOTO_BUCKET = "inspection-photos"

PPE_CLASSES = [
    "helmet",
    "gloves",
    "safety_vest",
    "safety_shoes",
    "harness",
]

PPE_LABELS = {
    "helmet": "Safety helmet",
    "gloves": "Safety gloves",
    "safety_vest": "Reflective safety vest",
    "safety_shoes": "Safety boots",
    "harness": "Full-body safety harness",
}

PPE_CONFIDENCE = 0.70

# Change this if your trained model uses different class names.
CLASS_ALIASES = {
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "gloves": "gloves",
    "glove": "gloves",
    "safety_gloves": "gloves",
    "safety_vest": "safety_vest",
    "vest": "safety_vest",
    "reflective_vest": "safety_vest",
    "safety_shoes": "safety_shoes",
    "boots": "safety_shoes",
    "safety_boots": "safety_shoes",
    "harness": "harness",
    "safety_harness": "harness",
    "full_body_harness": "harness",
}


# ============================================================
# 2. INITIALIZE SESSION STATE
# ============================================================

DEFAULT_SESSION_VALUES = {
    "supabase_client": None,
    "user": None,
    "active_inspection": None,
    "front_capture": None,
    "back_capture": None,
    "left_capture": None,
    "right_capture": None,
}

for key, default_value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# 3. SUPABASE CONNECTION
# ============================================================

def make_client():
    """
    Connect to Supabase using Streamlit secrets or environment
    variables.

    Streamlit secrets should contain:
        SUPABASE_URL = "https://your-project.supabase.co"
        SUPABASE_KEY = "your-key"
    """

    if st.session_state.supabase_client is not None:
        return st.session_state.supabase_client

    try:
        supabase_url = st.secrets.get(
            "SUPABASE_URL",
            os.getenv("SUPABASE_URL", ""),
        )

        supabase_key = st.secrets.get(
            "SUPABASE_KEY",
            os.getenv("SUPABASE_KEY", ""),
        )

        if not supabase_url or not supabase_key:
            st.error(
                "Supabase credentials are missing. "
                "Configure SUPABASE_URL and SUPABASE_KEY "
                "in Streamlit secrets."
            )
            return None

        client = create_client(
            supabase_url,
            supabase_key,
        )

        st.session_state.supabase_client = client

        return client

    except Exception as exc:
        st.error(f"Could not connect to Supabase: {exc}")
        return None


# ============================================================
# 4. USER INFORMATION HELPERS
# ============================================================

def get_user():
    """Return the current authenticated user."""

    return st.session_state.get("user")


def get_user_id():
    """Return the authenticated user's ID."""

    user = get_user()

    if user is None:
        return None

    return getattr(user, "id", None)


def get_user_email():
    """Return the authenticated user's email."""

    user = get_user()

    if user is None:
        return ""

    return getattr(user, "email", "") or ""


def get_user_role(client):
    """
    Get the current user's role from the profiles table.

    Expected columns:
        id
        role

    Example roles:
        worker
        inspector
        admin
    """

    user_id = get_user_id()

    if not user_id:
        return "worker"

    try:
        response = (
            client.table("profiles")
            .select("role")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )

        rows = response.data or []

        if rows:
            role = str(rows[0].get("role", "worker")).lower()

            if role in {"worker", "inspector", "admin"}:
                return role

    except Exception:
        pass

    return "worker"


# ============================================================
# 5. FETCH RWANDA PROVINCE, DISTRICT, AND SECTOR DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rwanda_locations(_client):
    """
    Fetch Rwanda's administrative locations from Supabase.

    Expected table:
        rwanda_sectors

    Expected columns:
        province
        district
        sector

    Returns a DataFrame containing these three columns.
    """

    columns = [
        "province",
        "district",
        "sector",
    ]

    try:
        response = (
            _client.table("rwanda_sectors")
            .select("province,district,sector")
            .execute()
        )

        rows = response.data or []

        if not rows:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(rows)

        for column in columns:
            if column not in df.columns:
                df[column] = ""

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        df = df[
            (df["province"] != "")
            & (df["district"] != "")
            & (df["sector"] != "")
        ]

        df = df.drop_duplicates(
            subset=columns
        ).sort_values(
            columns
        ).reset_index(drop=True)

        return df

    except Exception as exc:
        st.error(
            "Could not load Rwanda's administrative locations. "
            f"Check the rwanda_sectors table. Details: {exc}"
        )

        return pd.DataFrame(columns=columns)


# ============================================================
# 6. YOLOV8 PPE MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading PPE detection model...")
def load_ppe_model():
    """
    Load the trained YOLOv8 model.

    The best.pt file should be located in the same directory
    as app.py unless YOLO_MODEL_PATH is configured.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}. "
            "Place best.pt in the application directory."
        )

    return YOLO(str(MODEL_PATH))


# ============================================================
# 7. IMAGE ANALYSIS
# ============================================================

def analyze_image(image_bytes, confidence=PPE_CONFIDENCE):
    """
    Analyze an image using YOLOv8.

    Returns:
        {
            "success": bool,
            "detections": list,
            "counts": dict,
            "ppe_status": str,
            "error": str
        }
    """

    result_data = {
        "success": False,
        "detections": [],
        "counts": {
            item: 0 for item in PPE_CLASSES
        },
        "ppe_status": "Unable to analyze",
        "error": None,
    }

    try:
        if not image_bytes:
            result_data["error"] = "No image was provided."
            return result_data

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            result_data["error"] = (
                "The uploaded image could not be decoded."
            )
            return result_data

        model = load_ppe_model()

        predictions = model.predict(
            source=image,
            imgsz=800,
            conf=confidence,
            verbose=False,
        )

        prediction = predictions[0]

        class_names = prediction.names

        for box in prediction.boxes:
            class_id = int(
                box.cls[0].item()
            )

            score = float(
                box.conf[0].item()
            )

            original_name = str(
                class_names[class_id]
            ).strip().lower()

            normalized_name = (
                original_name
                .replace(" ", "_")
                .replace("-", "_")
            )

            ppe_class = CLASS_ALIASES.get(
                normalized_name
            )

            # Ignore classes that are not recognized PPE items.
            if ppe_class not in PPE_CLASSES:
                continue

            coordinates = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )

            result_data["counts"][ppe_class] += 1

            result_data["detections"].append({
                "class": ppe_class,
                "label": PPE_LABELS[ppe_class],
                "confidence": round(score, 4),
                "box": [
                    round(float(value), 2)
                    for value in coordinates
                ],
            })

        detected_items = [
            item
            for item in PPE_CLASSES
            if result_data["counts"][item] > 0
        ]

        missing_items = [
            PPE_LABELS[item]
            for item in PPE_CLASSES
            if result_data["counts"][item] == 0
        ]

        if not detected_items:
            status = "No required PPE detected"

        elif not missing_items:
            status = "All required PPE classes detected"

        else:
            status = "Some required PPE classes not detected"

        result_data["success"] = True
        result_data["ppe_status"] = status
        result_data["missing_items"] = missing_items

        return result_data

    except Exception as exc:
        result_data["error"] = str(exc)
        return result_data


# ============================================================
# 8. PREPARE ANALYSIS FOR DATABASE STORAGE
# ============================================================

def analysis_for_database(analysis):
    """
    Convert analysis results into a JSON-compatible dictionary.
    """

    if not analysis:
        return {
            "success": False,
            "error": "No analysis results available.",
        }

    return {
        "success": bool(
            analysis.get("success", False)
        ),
        "ppe_status": analysis.get(
            "ppe_status",
            "Unable to analyze",
        ),
        "counts": analysis.get(
            "counts",
            {},
        ),
        "detections": analysis.get(
            "detections",
            [],
        ),
        "missing_items": analysis.get(
            "missing_items",
            [],
        ),
        "error": analysis.get("error"),
    }


# ============================================================
# 9. PRIVATE PHOTO UPLOAD
# ============================================================

def upload_private_photo(
    client,
    image_bytes,
    inspection_code,
    view_name,
):
    """
    Upload an inspection photo to the private Supabase bucket.

    Returns the uploaded object's storage path, or None.
    """

    if not image_bytes:
        return None

    safe_view = (
        str(view_name)
        .strip()
        .lower()
        .replace(" ", "_")
    )

    storage_path = (
        f"{inspection_code}/"
        f"{safe_view}_{uuid.uuid4().hex}.jpg"
    )

    try:
        client.storage.from_(PHOTO_BUCKET).upload(
            path=storage_path,
            file=image_bytes,
            file_options={
                "content-type": "image/jpeg",
                "upsert": "false",
            },
        )

        return storage_path

    except Exception as exc:
        st.error(
            f"Could not upload {view_name} photo: {exc}"
        )

        return None


# ============================================================
# 10. READ A PRIVATE PHOTO
# ============================================================

def show_private_photo(
    client,
    storage_path,
    caption="Inspection photo",
):
    """
    Download and display a photo from private Supabase storage.

    The bucket must remain private.
    """

    if not storage_path:
        st.info("No photo is available.")
        return

    try:
        image_bytes = (
            client.storage
            .from_(PHOTO_BUCKET)
            .download(storage_path)
        )

        st.image(
            image_bytes,
            caption=caption,
            use_container_width=True,
        )

    except Exception as exc:
        st.warning(
            f"Could not display the photo: {exc}"
        )


# ============================================================
# 11. FETCH INSPECTIONS
# ============================================================

def fetch_inspections(client, limit=200):
    """
    Retrieve inspection records.

    Row-level security must restrict each user's access
    to authorized records.
    """

    try:
        response = (
            client.table("inspections")
            .select("*")
            .order(
                "created_at",
                desc=True,
            )
            .limit(limit)
            .execute()
        )

        return response.data or []

    except Exception as exc:
        st.error(
            f"Could not retrieve inspections: {exc}"
        )

        return []


# ============================================================
# 12. FETCH USER NOTIFICATIONS
# ============================================================

def fetch_notifications(client):
    """Fetch notifications for the authenticated user."""

    user_id = get_user_id()

    if not user_id:
        return []

    try:
        response = (
            client.table("notifications")
            .select("*")
            .eq("user_id", user_id)
            .order(
                "created_at",
                desc=True,
            )
            .limit(100)
            .execute()
        )

        return response.data or []

    except Exception:
        return []


# ============================================================
# 13. INSPECTION STATUS DISPLAY
# ============================================================

def display_status(status):
    """Display an inspection status using a suitable message."""

    normalized = str(
        status or "pending"
    ).strip().lower()

    if normalized in {"approved", "accepted"}:
        st.success(f"Status: {status}")

    elif normalized in {"rejected", "declined"}:
        st.error(f"Status: {status}")

    elif normalized in {
        "pending",
        "under review",
        "under_review",
    }:
        st.warning(f"Status: {status}")

    else:
        st.info(f"Status: {status}")


# ============================================================
# 14. COMMON DISPLAY HELPERS
# ============================================================

def display_ppe_results(analysis):
    """Display PPE detection results in a table."""

    if not analysis:
        st.info("No PPE analysis is available.")
        return

    counts = analysis.get("counts", {})

    rows = []

    for ppe_class in PPE_CLASSES:
        rows.append({
            "Required PPE": PPE_LABELS[ppe_class],
            "Detections": counts.get(ppe_class, 0),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "**Detection result:**",
        analysis.get(
            "ppe_status",
            "Unable to analyze",
        ),
    )

    missing = analysis.get("missing_items", [])

    if missing:
        st.warning(
            "Not detected: " + ", ".join(missing)
        )

    if analysis.get("error"):
        st.error(analysis["error"])


def current_timestamp():
    """Return a UTC timestamp in ISO 8601 format."""

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# END OF PART 1
# ============================================================
# ============================================================
# PART 2: TOWERSAFE PPE INSPECTION APPLICATION
# ============================================================

import json
import traceback


# ------------------------------------------------------------
# AUTHENTICATION
# ------------------------------------------------------------

def towersafe_logout():
    """Log out the current user."""
    client = st.session_state.get("supabase_client")

    try:
        if client is not None:
            client.auth.sign_out()
    except Exception:
        pass

    for key in [
        "user",
        "active_inspection",
        "front_capture",
        "back_capture",
        "left_capture",
        "right_capture",
        "access_token",
        "refresh_token",
    ]:
        st.session_state.pop(key, None)

    st.rerun()


def towersafe_authentication_page(client):
    """Display the registration and login page."""

    st.title("TowerSafe AI")
    st.subheader("Telecom Tower PPE Inspection System")

    st.write(
        "Register as a worker or log in to access your dashboard."
    )

    login_tab, register_tab = st.tabs(
        ["Login", "Register as a Worker"]
    )

    with login_tab:
        st.markdown("### Welcome back")

        login_email = st.text_input(
            "Email address",
            key="towersafe_login_email",
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="towersafe_login_password",
        )

        if st.button(
            "Log in",
            type="primary",
            use_container_width=True,
        ):
            if not login_email.strip() or not login_password:
                st.error("Enter your email and password.")
            else:
                try:
                    response = client.auth.sign_in_with_password(
                        {
                            "email": login_email.strip(),
                            "password": login_password,
                        }
                    )

                    user = response.user
                    session = response.session

                    if user is None:
                        st.error("Login failed. Please try again.")
                    else:
                        st.session_state.user = user

                        if session is not None:
                            st.session_state.access_token = (
                                session.access_token
                            )
                            st.session_state.refresh_token = (
                                session.refresh_token
                            )

                        st.success("Login successful!")
                        st.rerun()

                except Exception as exc:
                    st.error(
                        "Login failed. Check your email and password."
                    )
                    st.caption(str(exc))

    with register_tab:
        st.markdown("### Create a worker account")

        register_name = st.text_input(
            "Full name",
            key="towersafe_register_name",
        )

        register_email = st.text_input(
            "Email address",
            key="towersafe_register_email",
        )

        register_password = st.text_input(
            "Create password",
            type="password",
            key="towersafe_register_password",
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="towersafe_confirm_password",
        )

        st.info(
            "New accounts are for workers. Inspector and admin "
            "access must be assigned separately."
        )

        if st.button(
            "Register",
            type="primary",
            use_container_width=True,
        ):
            if not register_name.strip():
                st.error("Enter your full name.")

            elif not register_email.strip():
                st.error("Enter your email address.")

            elif len(register_password) < 6:
                st.error(
                    "Your password must contain at least 6 characters."
                )

            elif register_password != confirm_password:
                st.error("The passwords do not match.")

            else:
                try:
                    response = client.auth.sign_up(
                        {
                            "email": register_email.strip(),
                            "password": register_password,
                            "options": {
                                "data": {
                                    "full_name": register_name.strip()
                                }
                            },
                        }
                    )

                    if response.user is None:
                        st.error(
                            "Registration failed. Please try again."
                        )

                    elif response.session is None:
                        st.success(
                            "Account created. Check your email "
                            "to confirm your account, then log in."
                        )

                    else:
                        st.session_state.user = response.user
                        st.session_state.access_token = (
                            response.session.access_token
                        )
                        st.session_state.refresh_token = (
                            response.session.refresh_token
                        )

                        st.success("Your worker account was created.")
                        st.rerun()

                except Exception as exc:
                    st.error("Registration failed.")
                    st.caption(str(exc))


# ------------------------------------------------------------
# RESTORE AUTHENTICATION AFTER STREAMLIT RERUNS
# ------------------------------------------------------------

def towersafe_restore_session(client):
    """Restore the saved Supabase authentication session."""

    access_token = st.session_state.get("access_token")
    refresh_token = st.session_state.get("refresh_token")

    if access_token and refresh_token:
        try:
            response = client.auth.set_session(
                access_token,
                refresh_token,
            )

            if response.user is not None:
                st.session_state.user = response.user
                return response.user

        except Exception:
            st.session_state.pop("user", None)
            st.session_state.pop("access_token", None)
            st.session_state.pop("refresh_token", None)

    user = st.session_state.get("user")

    if user is not None:
        return user

    return None


# ------------------------------------------------------------
# WORKER INSPECTION SUBMISSION
# ------------------------------------------------------------

def towersafe_submit_inspection(
    client,
    technician_name,
    site_id,
    site_name,
    photo_bytes,
):
    """
    Analyze four photos, upload the photos, and save an inspection.

    Front and back photo paths use their own database columns.
    Left and right photo paths and analysis details are stored
    in the JSON-compatible ppe_results column.
    """

    inspection_code = (
        "TS-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        + "-" + uuid.uuid4().hex[:6].upper()
    )

    analyses = {}
    uploaded_paths = {}

    # Analyze each view.
    for view_name in ["front", "back", "left", "right"]:
        result = analyze_image(
            photo_bytes[view_name],
            confidence=PPE_CONFIDENCE,
        )

        if not result.get("success"):
            raise ValueError(
                f"AI analysis failed for the {view_name} view: "
                f"{result.get('error', 'Unknown error')}"
            )

        analyses[view_name] = analysis_for_database(result)

    # Upload all four photos to private Supabase storage.
    for view_name in ["front", "back", "left", "right"]:
        uploaded_paths[view_name] = upload_private_photo(
            client,
            photo_bytes[view_name],
            inspection_code,
            view_name,
        )

    # Combine detections from the four views.
    detected_classes = set()

    for result in analyses.values():
        counts = result.get("counts", {})

        for class_name, count in counts.items():
            if count and int(count) > 0:
                detected_classes.add(class_name)

    missing_classes = [
        class_name
        for class_name in PPE_CLASSES
        if class_name not in detected_classes
    ]

    if not missing_classes:
        overall_ppe_status = "All required PPE classes detected"
    elif len(missing_classes) == len(PPE_CLASSES):
        overall_ppe_status = "No required PPE detected"
    else:
        overall_ppe_status = "Some required PPE classes not detected"

    user_id = get_user_id()

    if not user_id:
        raise ValueError(
            "Your user ID could not be identified. Please log in again."
        )

    # Keep the four-view details in ppe_results.
    ppe_results = {
        "overall_status": overall_ppe_status,
        "required_ppe": PPE_CLASSES,
        "missing_items": missing_classes,
        "photo_paths": uploaded_paths,
        "views": analyses,
    }

    # Insert only the confirmed database columns.
    inspection_row = {
        "inspection_code": inspection_code,
        "technician_name": technician_name.strip(),
        "site_id": site_id.strip(),
        "site_name": site_name.strip(),
        "worker_user_id": str(user_id),
        "status": "UNDER REVIEW",
        "front_photo_path": uploaded_paths["front"],
        "back_photo_path": uploaded_paths["back"],
        "ppe_results": ppe_results,
    }

    response = (
        client.table("inspections")
        .insert(inspection_row)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Supabase did not return the saved inspection record."
        )

    return response.data[0]


# ------------------------------------------------------------
# WORKER DASHBOARD
# ------------------------------------------------------------

def towersafe_worker_dashboard(client):

    st.title("Worker Dashboard")
    st.write(
        "Capture four views of the worker for AI-assisted "
        "personal protective equipment inspection."
    )

    st.warning(
        "Safety reminder: AI results support inspection. "
        "They do not replace a qualified safety inspection."
    )

    user = st.session_state.get("user")
    user_metadata = getattr(user, "user_metadata", None) or {}

    default_name = user_metadata.get("full_name", "")

    with st.form("towersafe_inspection_details"):
        st.subheader("1. Inspection details")

        technician_name = st.text_input(
            "Worker's full name",
            value=default_name,
        )

        site_id = st.text_input("Site ID")

        site_name = st.text_input(
            "Telecom tower site name or location"
        )

        st.subheader("2. Capture the four views")

        st.caption(
            "Take clear photos with the worker fully visible. "
            "Use the same worker for all four views."
        )

        front_col, back_col = st.columns(2)

        with front_col:
            st.markdown("#### Front view")
            front = st.camera_input(
                "Take front-view photo",
                key="front_camera",
            )

        with back_col:
            st.markdown("#### Back view")
            back = st.camera_input(
                "Take back-view photo",
                key="back_camera",
            )

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### Left-side view")
            left = st.camera_input(
                "Take left-side photo",
                key="left_camera",
            )

        with right_col:
            st.markdown("#### Right-side view")
            right = st.camera_input(
                "Take right-side photo",
                key="right_camera",
            )

        submitted = st.form_submit_button(
            "Submit evidence to inspector",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if not technician_name.strip():
            st.error("Enter the worker's name.")

        elif not site_id.strip():
            st.error("Enter the site ID.")

        elif not site_name.strip():
            st.error("Enter the tower site name or location.")

        elif any(
            photo is None
            for photo in [front, back, left, right]
        ):
            st.error(
                "Please capture all four views: "
                "front, back, left, and right."
            )

        else:
            photo_bytes = {
                "front": front.getvalue(),
                "back": back.getvalue(),
                "left": left.getvalue(),
                "right": right.getvalue(),
            }

            try:
                with st.spinner(
                    "Analyzing PPE and submitting evidence..."
                ):
                    saved_inspection = towersafe_submit_inspection(
                        client=client,
                        technician_name=technician_name,
                        site_id=site_id,
                        site_name=site_name,
                        photo_bytes=photo_bytes,
                    )

                st.success(
                    "Inspection submitted successfully!"
                )

                st.write(
                    "Inspection code:",
                    saved_inspection["inspection_code"],
                )

                st.info(
                    "Your evidence has been submitted for inspector review."
                )

            except Exception as exc:
                st.error("The inspection could not be submitted.")
                st.write(str(exc))

                st.caption(
                    "If the error mentions storage, check that the "
                    "inspection-photos bucket exists and that the "
                    "signed-in worker has permission to upload."
                )


# ------------------------------------------------------------
# INSPECTION HISTORY
# ------------------------------------------------------------

def towersafe_inspection_history(client):

    st.title("Inspection History")

    user_id = get_user_id()

    if not user_id:
        st.error("Please log in again.")
        return

    try:
        response = (
            client.table("inspections")
            .select("*")
            .eq("worker_user_id", str(user_id))
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        records = response.data or []

        if not records:
            st.info("You have not submitted any inspections yet.")
            return

        display_records = []

        for record in records:
            display_records.append({
                "Inspection code": record.get("inspection_code"),
                "Worker": record.get("technician_name"),
                "Site": record.get("site_name"),
                "Status": record.get("status"),
                "Created": record.get("created_at"),
            })

        st.dataframe(
            pd.DataFrame(display_records),
            use_container_width=True,
            hide_index=True,
        )

        selected_code = st.selectbox(
            "Select an inspection to view its details",
            options=[
                record.get("inspection_code")
                for record in records
            ],
        )

        selected_record = next(
            (
                record
                for record in records
                if record.get("inspection_code") == selected_code
            ),
            None,
        )

        if selected_record:
            st.json(selected_record.get("ppe_results") or {})

    except Exception as exc:
        st.error("Could not load inspection history.")
        st.caption(str(exc))


# ------------------------------------------------------------
# INSPECTOR DASHBOARD
# ------------------------------------------------------------

def towersafe_inspector_dashboard(client):

    st.title("Inspector Dashboard")

    st.write(
        "Review submitted evidence and record your decision."
    )

    try:
        response = (
            client.table("inspections")
            .select("*")
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )

        records = response.data or []

    except Exception as exc:
        st.error("Could not load inspections.")
        st.caption(str(exc))
        return

    if not records:
        st.info("There are no inspections to display.")
        return

    pending = [
        record for record in records
        if str(record.get("status", "")).lower() == "pending"
    ]

    st.metric("Total inspections loaded", len(records))
    st.metric("Pending review", len(pending))

    st.subheader("Inspection records")

    summary = []

    for record in records:
        summary.append({
            "Inspection code": record.get("inspection_code"),
            "Worker": record.get("technician_name"),
            "Site": record.get("site_name"),
            "Status": record.get("status"),
            "Created": record.get("created_at"),
        })

    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Review evidence")

    codes = [
        record.get("inspection_code")
        for record in records
        if record.get("inspection_code")
    ]

    selected_code = st.selectbox(
        "Choose an inspection",
        options=codes,
    )

    record = next(
        (
            item for item in records
            if item.get("inspection_code") == selected_code
        ),
        None,
    )

    if record is None:
        return

    st.write("**Worker:**", record.get("technician_name"))
    st.write("**Site:**", record.get("site_name"))
    st.write("**Current status:**", record.get("status"))

    st.markdown("### AI PPE analysis")

    ppe_results = record.get("ppe_results") or {}

    if isinstance(ppe_results, str):
        try:
            ppe_results = json.loads(ppe_results)
        except Exception:
            ppe_results = {}

    st.json(ppe_results)

    st.markdown("### Captured photographs")

    photo_paths = ppe_results.get("photo_paths", {})

    if not isinstance(photo_paths, dict):
        photo_paths = {}

    photo_paths.setdefault(
        "front",
        record.get("front_photo_path"),
    )

    photo_paths.setdefault(
        "back",
        record.get("back_photo_path"),
    )

    photo_columns = st.columns(2)

    for index, view_name in enumerate(
        ["front", "back", "left", "right"]
    ):
        storage_path = photo_paths.get(view_name)

        with photo_columns[index % 2]:
            st.markdown(f"#### {view_name.title()} view")

            if storage_path:
                try:
                    show_private_photo(
                        client,
                        storage_path,
                        f"{view_name.title()} view",
                    )
                except Exception as exc:
                    st.warning(
                        f"Could not display this photograph: {exc}"
                    )
            else:
                st.info("No photograph path is available.")

    st.markdown("### Inspector decision")

    current_status = str(record.get("status", "")).lower()

    if current_status == "pending":

        decision = st.radio(
            "Decision",
            ["Approve", "Reject"],
            horizontal=True,
            key=f"decision_{selected_code}",
        )

        if st.button(
            "Save inspector decision",
            type="primary",
            key=f"save_decision_{selected_code}",
        ):
            new_status = (
                "approved"
                if decision == "Approve"
                else "rejected"
            )

            try:
                (
                    client.table("inspections")
                    .update({"status": new_status})
                    .eq("inspection_code", selected_code)
                    .execute()
                )

                st.success(
                    f"Inspection marked as {new_status}."
                )

                st.rerun()

            except Exception as exc:
                st.error("Could not save the inspector decision.")
                st.caption(str(exc))

    else:
        st.info(
            f"This inspection has already been marked "
            f"'{record.get('status')}'."
        )


# ------------------------------------------------------------
# MAIN APPLICATION
# ------------------------------------------------------------

def towersafe_main():

    st.set_page_config(
        page_title="TowerSafe AI",
        page_icon="🦺",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Connect to Supabase using the Part 1 helper.
    try:
        client = make_client()

    except Exception as exc:
        st.title("TowerSafe AI")
        st.error("Could not connect to Supabase.")
        st.write(str(exc))
        st.stop()

    if client is None:
        st.title("TowerSafe AI")
        st.error(
            "Supabase is not configured. Check SUPABASE_URL "
            "and SUPABASE_KEY in Streamlit secrets."
        )
        st.stop()

    user = towersafe_restore_session(client)

    if user is None:
        towersafe_authentication_page(client)
        return

    try:
        role = get_user_role(client)
    except Exception:
        role = "worker"

    role = str(role or "worker").lower()

    st.sidebar.title("🦺 TowerSafe AI")

    st.sidebar.write(
        getattr(user, "email", "Signed-in user")
    )

    st.sidebar.write(f"Account role: **{role}**")

    if st.sidebar.button(
        "Log out",
        use_container_width=True,
    ):
        towersafe_logout()

    if role in ["inspector", "admin"]:

        page = st.sidebar.radio(
            "Navigation",
            [
                "Inspector Dashboard",
                "Worker Dashboard",
                "Inspection History",
            ],
        )

    else:
        page = st.sidebar.radio(
            "Navigation",
            [
                "Worker Dashboard",
                "Inspection History",
            ],
        )

    if page == "Worker Dashboard":
        towersafe_worker_dashboard(client)

    elif page == "Inspection History":
        towersafe_inspection_history(client)

    elif page == "Inspector Dashboard":
        if role in ["inspector", "admin"]:
            towersafe_inspector_dashboard(client)
        else:
            st.error("You are not authorized to access this page.")


# Start the Streamlit application.
towersafe_main()








