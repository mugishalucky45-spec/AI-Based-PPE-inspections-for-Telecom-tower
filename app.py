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