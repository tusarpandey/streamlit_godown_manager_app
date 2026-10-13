"""Godown Manager - Streamlit web UI.

Same backend as the Tkinter app (core/), reused unmodified. The one real
architecture change is the camera: Tkinter drove a live local webcam with a
30ms polling loop and multi-frame liveness/gesture checks; Streamlit's
st.camera_input is snapshot-only (one photo per click, taken in the
browser), so every camera-driven flow here (gatekeeper login, enrollment,
OCR capture) works off a single photo instead of a continuous stream.
"""

import threading

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_card import card

from core import PROJECT_ROOT
from core.auth_engine import AuthEngine
from core.inventory_db import InventoryDB
from core.model_manager import ModelManager
from core.ocr_engine import OCREngine
from core.profile_store import ProfileStore

try:
    ADMIN_PIN = st.secrets.get("ADMIN_PIN", "1234")
except Exception:
    ADMIN_PIN = "1234"

UNITS = ["Nos", "Kg", "Gms", "Ltr", "Ml", "Box", "Pcs", "Mtr"]

# Demo rows written to data/inventory.xlsx the first time it's created (e.g.
# a fresh clone, or a cloud deploy with an ephemeral filesystem that reset
# on restart), so Check Inventory / Item Detail have something to show
# instead of an empty sheet.
DEMO_ITEMS = [
    ("Rice Bags 25kg", 40, "Box"),
    ("Cement 50kg", 120, "Box"),
    ("Wheat Flour", 85, "Kg"),
    ("Cooking Oil 15L", 30, "Ltr"),
    ("Steel Rods 12mm", 200, "Pcs"),
]

st.set_page_config(page_title="Godown Manager", page_icon="\U0001F4E6", layout="centered")


# ---------------------------------------------------------------- singletons

@st.cache_resource
def get_profile_store():
    return ProfileStore()


@st.cache_resource
def get_inventory_db():
    path = PROJECT_ROOT / "data" / "inventory.xlsx"
    is_new = not path.exists()
    db = InventoryDB()
    if is_new:
        for desc, qty, unit in DEMO_ITEMS:
            db.keep_item(desc, qty, unit, manager_name="Seed")
    return db


@st.cache_resource
def get_model_manager():
    auth_engine = AuthEngine(get_profile_store())
    ocr_engine = OCREngine()
    return ModelManager(auth_engine, ocr_engine)


@st.cache_resource
def get_inference_lock():
    """get_model_manager() is one shared instance across every session on
    this server (st.cache_resource is process-wide), and ModelManager keeps
    only one heavy model (face vs OCR) loaded at a time - so two users
    snapshotting at once could otherwise interleave swap-and-infer calls on
    the same underlying model. Every inference call below is serialized
    through this single lock."""
    return threading.Lock()


# ------------------------------------------------------------------ helpers

def snapshot_to_numpy(camera_file):
    """st.camera_input() hands back a BytesIO-like UploadedFile (JPEG); the
    core/ engines expect a plain BGR NumPy array, same as a cv2.VideoCapture
    frame."""
    if camera_file is None:
        return None
    image = Image.open(camera_file).convert("RGB")
    frame = np.array(image)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _is_new_photo(photo, state_key):
    """True once per distinct captured photo. st.camera_input keeps
    returning the same photo on every rerun until it's retaken/cleared, so
    without this guard a heavy model (face match / OCR) would re-run on
    every unrelated widget interaction on the same page."""
    if photo is None:
        return False
    data = photo.getvalue()
    if st.session_state.get(state_key) == data:
        return False
    st.session_state[state_key] = data
    return True


def goto(page):
    st.session_state.page = page
    st.rerun()


def _topbar(title, back_to=None):
    if back_to:
        left, right = st.columns([4, 1])
        left.subheader(title)
        if right.button("⬅ Back", key=f"back_{back_to}_{title}"):
            goto(back_to)
    else:
        st.subheader(title)


def _init_session_state():
    st.session_state.setdefault("page", "gatekeeper")
    st.session_state.setdefault("current_manager_name", None)
    st.session_state.setdefault("selected_item", None)
    st.session_state.setdefault("new_item_mode", False)


# -------------------------------------------------------------------- pages

def page_gatekeeper():
    st.title("\U0001F4E6 Godown Manager")
    st.subheader("Manager Verification")

    profile_store = get_profile_store()
    profile_store.reload()

    if profile_store.is_empty:
        st.info("No one enrolled yet. Continue to the dashboard so you can enroll the first person.")
        if st.button("Continue to Dashboard", type="primary", width="stretch"):
            goto("dashboard")
        return

    st.write("Take a photo to verify your identity.")
    photo = st.camera_input("Look at the camera", key="gatekeeper_cam")

    if photo is None:
        return

    frame = snapshot_to_numpy(photo)
    with st.spinner("Verifying..."):
        with get_inference_lock():
            auth_engine = get_model_manager().get_auth()
            result = auth_engine.verify(frame)

    if not result.face_found:
        st.error("No face detected. Retake the photo with better lighting.")
    elif result.is_live is False:
        st.error("Liveness check failed - move closer / improve lighting, then retake the photo.")
    elif result.matched:
        st.session_state.current_manager_name = result.matched_name
        goto("dashboard")
    else:
        pct = int((result.similarity or 0) * 100)
        st.error(f"Face not recognized ({pct}% match). Retake the photo or ask an admin to enroll you.")


# Dashboard nav cards: (key, title, text, gradient, target_page, admin_redirect).
# Enroll and Admin Panel both land on the PIN screen first (admin_redirect),
# which then forwards to the real destination on success - Enroll keeps the
# same PIN gate the Tkinter app had, just reachable in one tap instead of via
# the Admin Panel submenu.
DASHBOARD_CARDS = [
    {
        "key": "nav_check", "title": "\U0001F50D Check Inventory", "text": "Search & OCR scan",
        "gradient": "linear-gradient(135deg, #2563eb, #1d4ed8)", "target_page": "check",
    },
    {
        "key": "nav_keep", "title": "➕ Add / Withdraw", "text": "Stock updates",
        "gradient": "linear-gradient(135deg, #22c55e, #d97706)", "target_page": "keep",
    },
    {
        "key": "nav_enroll", "title": "\U0001F464 Enroll Manager", "text": "Face profile registry",
        "gradient": "linear-gradient(135deg, #7c3aed, #4f46e5)", "admin_redirect": "enroll",
    },
    {
        "key": "nav_admin", "title": "\U0001F510 Admin Panel", "text": "Settings & history logs",
        "gradient": "linear-gradient(135deg, #243b55, #1b2c40)", "admin_redirect": "admin_panel",
    },
]


def _nav_card(title, text, gradient, key, target_page=None, admin_redirect=None):
    clicked = card(
        title=title,
        text=text,
        image=None,
        styles={
            "card": {
                "width": "100%",
                "height": "160px",
                "borderRadius": "16px",
                "boxShadow": "0 6px 18px rgba(0, 0, 0, 0.25)",
                "margin": "6px 0px",
                "background": gradient,
            },
            "filter": {"backgroundColor": "rgba(0, 0, 0, 0.35)"},
        },
        key=key,
    )
    if clicked:
        if admin_redirect:
            st.session_state["admin_login_redirect"] = admin_redirect
            goto("admin_login")
        else:
            goto(target_page)


def page_dashboard():
    st.title("\U0001F4E6 Godown Manager")
    name = st.session_state.current_manager_name
    if name:
        st.caption(f"Signed in as **{name}**")

    row1 = st.columns(2)
    row2 = st.columns(2)
    for col, spec in zip(row1 + row2, DASHBOARD_CARDS):
        with col:
            _nav_card(
                spec["title"], spec["text"], spec["gradient"], spec["key"],
                target_page=spec.get("target_page"), admin_redirect=spec.get("admin_redirect"),
            )

    st.divider()
    if st.button("Logout"):
        st.session_state.current_manager_name = None
        goto("gatekeeper")


def page_admin_login():
    _topbar("Admin Login", back_to="dashboard")

    with st.form("admin_login_form", clear_on_submit=True):
        pin = st.text_input("Enter Admin PIN", type="password")
        submitted = st.form_submit_button("Unlock", width="stretch")

    if submitted:
        if pin == ADMIN_PIN:
            goto(st.session_state.pop("admin_login_redirect", "admin_panel"))
        else:
            st.error("Incorrect PIN")


def page_admin_panel():
    _topbar("Admin", back_to="dashboard")

    if st.button("\U0001F464 Enroll New Person", type="primary", width="stretch"):
        goto("enroll")

    st.markdown("**Enrolled people:**")
    profile_store = get_profile_store()
    profile_store.reload()
    names = profile_store.names
    st.write(", ".join(names) if names else "_(none yet)_")

    if names:
        with st.expander("Remove a person"):
            remove_name = st.selectbox("Select", names, key="remove_name_select")
            if st.button(f"Remove {remove_name}"):
                profile_store.remove(remove_name)
                st.success(f"Removed {remove_name}.")
                st.rerun()


def _get_enroll_face(photo):
    """Runs face detection + liveness once per distinct captured photo and
    caches the outcome, so re-renders (e.g. typing in the name field) don't
    re-run the heavy model."""
    if photo is None:
        st.session_state.pop("_enroll_face_cache", None)
        return None

    data = photo.getvalue()
    cache = st.session_state.get("_enroll_face_cache")
    if cache and cache["bytes"] == data:
        return cache["result"]

    frame = snapshot_to_numpy(photo)
    with get_inference_lock():
        auth_engine = get_model_manager().get_auth()
        face = auth_engine.detect(frame)

    if face is None:
        result = {"ok": False, "msg": "No face detected. Retake the photo."}
    else:
        liveness = face.liveness
        is_live = liveness["is_live"] if liveness else True
        if not is_live:
            result = {"ok": False, "msg": "Liveness check failed. Retake with better lighting / move closer."}
        elif face.embedding is None:
            result = {"ok": False, "msg": "Could not extract a face embedding. Retake the photo."}
        else:
            result = {"ok": True, "embedding": face.embedding}

    st.session_state["_enroll_face_cache"] = {"bytes": data, "result": result}
    return result


def page_enroll():
    _topbar("Enroll New Person", back_to="admin_panel")

    st.write("Take a clear, front-facing photo with good lighting.")
    photo = st.camera_input("Capture face", key="enroll_cam")
    name = st.text_input("Name", key="enroll_name")

    face_result = _get_enroll_face(photo)
    if face_result is None:
        return
    if not face_result["ok"]:
        st.error(face_result["msg"])
        return

    st.success("Face captured.")
    if st.button("Save", type="primary", width="stretch", disabled=not name.strip()):
        get_profile_store().add(name.strip(), face_result["embedding"])
        st.session_state.pop("_enroll_face_cache", None)
        st.success(f"Saved '{name.strip()}'.")
        goto("admin_panel")


def page_check():
    _topbar("Check Inventory", back_to="dashboard")

    if st.session_state.pop("_reset_check", False):
        st.session_state["check_query"] = ""
        st.session_state.pop("check_results", None)
    st.session_state.setdefault("check_query", "")

    with st.expander("\U0001F4F7 Capture & OCR instead of typing"):
        photo = st.camera_input("Capture item label", key="check_cam")
        if photo is not None and _is_new_photo(photo, "_seen_check_photo"):
            frame = snapshot_to_numpy(photo)
            with st.spinner("Reading text..."):
                with get_inference_lock():
                    ocr = get_model_manager().get_ocr()
                    text = ocr.extract_text(frame)
            st.session_state["check_query"] = text
            st.rerun()

    query = st.text_input("Item description", key="check_query")

    col1, col2 = st.columns(2)
    with col1:
        search_clicked = st.button("\U0001F50D Search", type="primary", width="stretch")
    with col2:
        if st.button("\U0001F9F9 Clear", width="stretch"):
            st.session_state["_reset_check"] = True
            st.rerun()

    db = get_inventory_db()
    if search_clicked:
        if query.strip():
            st.session_state["check_results"] = db.search_item(query, limit=50)
        else:
            st.session_state.pop("check_results", None)

    results = st.session_state.get("check_results")

    if results is not None:
        if not results:
            st.warning("No matches found.")
        else:
            st.markdown(f"**{len(results)} match(es):**")
            for i, match in enumerate(results):
                desc = " ".join((match["Item_Description"] or "").split())
                label = f"{desc}  —  {match['Quantity']} {match['Unit']}"
                if st.button(label, key=f"result_{i}", width="stretch"):
                    st.session_state.selected_item = dict(match)
                    st.session_state.new_item_mode = False
                    goto("item_detail")

        if query.strip() and st.button("➕ Add as new item", width="stretch"):
            st.session_state.selected_item = {
                "Item_Description": query.strip(), "Quantity": 0, "Unit": "",
                "Timestamp": "", "Manager_Name": "",
            }
            st.session_state.new_item_mode = True
            goto("item_detail")


def page_item_detail():
    _topbar("Item Details", back_to="check")

    item = st.session_state.selected_item or {}
    is_new = st.session_state.new_item_mode
    desc = " ".join((item.get("Item_Description") or "").split())

    if not desc:
        st.warning("No item selected.")
        return

    status_msg = st.session_state.pop("item_status", None)

    st.subheader(desc)
    if status_msg:
        kind, text = status_msg
        getattr(st, kind)(text)

    db = get_inventory_db()

    if is_new:
        st.info("New item - not yet in inventory.")
        default_unit = item.get("Unit") if item.get("Unit") in UNITS else UNITS[0]
        unit = st.selectbox("Unit", UNITS, index=UNITS.index(default_unit))
    else:
        st.caption(
            f"Available: **{item.get('Quantity', 0)} {item.get('Unit', '')}**  ·  "
            f"last updated by {item.get('Manager_Name') or '-'} on {item.get('Timestamp') or '-'}"
        )
        unit = item.get("Unit", "Nos")

    if st.session_state.pop("_reset_item_qty", False):
        st.session_state["item_qty"] = 0.0
    qty = st.number_input("Quantity", min_value=0.0, step=1.0, key="item_qty")

    manager_name = st.session_state.current_manager_name
    withdraw_clicked = False

    if is_new:
        add_clicked = st.button("➕ Add", type="primary", width="stretch")
    else:
        col1, col2 = st.columns(2)
        with col1:
            add_clicked = st.button("➕ Add", type="primary", width="stretch")
        with col2:
            withdraw_clicked = st.button("➖ Withdraw", width="stretch")

    if add_clicked:
        if qty <= 0:
            st.error("Quantity is required.")
        else:
            try:
                result = db.keep_item(item["Item_Description"], qty, unit, manager_name=manager_name)
            except PermissionError:
                st.error("Close inventory.xlsx in Excel and try again.")
            except Exception as exc:
                st.error(f"Add failed: {exc}")
            else:
                item.update(Quantity=result["quantity"], Unit=unit,
                            Timestamp=result["timestamp"], Manager_Name=result["manager_entry"])
                st.session_state.selected_item = item
                st.session_state.new_item_mode = False
                st.session_state["item_status"] = ("success", f"Added: {result['description']} → qty {result['quantity']}")
                st.session_state["_reset_item_qty"] = True
                st.rerun()

    if withdraw_clicked:
        if qty <= 0:
            st.error("Quantity is required.")
        else:
            try:
                result = db.withdraw_item(item["Item_Description"], qty, manager_name=manager_name)
            except PermissionError:
                st.error("Close inventory.xlsx in Excel and try again.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Withdraw failed: {exc}")
            else:
                item.update(Quantity=result["quantity"], Timestamp=result["timestamp"],
                            Manager_Name=result["manager_entry"])
                st.session_state.selected_item = item
                st.session_state["item_status"] = (
                    "success", f"Withdrawn: {result['description']} → qty {result['quantity']} remaining")
                st.session_state["_reset_item_qty"] = True
                st.rerun()

    st.divider()
    st.markdown("**History**")
    history = db.get_history(desc)
    if not history:
        st.caption("No transactions yet.")
    else:
        for entry in history:
            mgr = entry["Manager_Name"] or "-"
            st.write(f"{mgr} on {entry['Timestamp']} → {entry['Resulting_Quantity']}")


def page_keep():
    """Direct add/withdraw with optional OCR capture - no search required.
    This screen existed fully built in the Tkinter app's code but was never
    wired to a button there; it's exposed here as a real dashboard option."""
    _topbar("Add / Withdraw Inventory", back_to="dashboard")

    if st.session_state.pop("_reset_keep_fields", False):
        st.session_state["keep_desc"] = ""
        st.session_state["keep_qty"] = 0.0

    status_msg = st.session_state.pop("keep_status", None)
    if status_msg:
        kind, text = status_msg
        getattr(st, kind)(text)

    with st.expander("\U0001F4F7 Capture & OCR"):
        photo = st.camera_input("Capture item label", key="keep_cam")
        if photo is not None and _is_new_photo(photo, "_seen_keep_photo"):
            frame = snapshot_to_numpy(photo)
            with st.spinner("Reading text..."):
                with get_inference_lock():
                    ocr = get_model_manager().get_ocr()
                    text = ocr.extract_text(frame)
            st.session_state["keep_desc"] = text
            st.rerun()

    st.session_state.setdefault("keep_desc", "")
    desc = st.text_area("Item description", key="keep_desc", height=100)

    col1, col2 = st.columns([2, 1])
    with col1:
        qty = st.number_input("Quantity", min_value=0.0, step=1.0, key="keep_qty")
    with col2:
        unit = st.selectbox("Unit", UNITS, key="keep_unit")

    manager_name = st.session_state.current_manager_name

    c1, c2, c3 = st.columns(3)
    with c1:
        reject_clicked = st.button("✖ Reject", width="stretch")
    with c2:
        add_clicked = st.button("➕ Add", type="primary", width="stretch")
    with c3:
        withdraw_clicked = st.button("➖ Withdraw", width="stretch")

    if reject_clicked:
        st.session_state["_reset_keep_fields"] = True
        st.rerun()

    db = get_inventory_db()

    if add_clicked or withdraw_clicked:
        if not desc.strip():
            st.error("Description is required.")
        elif qty <= 0:
            st.error("Quantity is required.")
        elif add_clicked:
            try:
                result = db.keep_item(desc, qty, unit, manager_name=manager_name)
            except PermissionError:
                st.error("Close inventory.xlsx in Excel and try again.")
            except Exception as exc:
                st.error(f"Add failed: {exc}")
            else:
                st.session_state["keep_status"] = (
                    "success", f"Added ({result['action']}): {result['description']} → qty {result['quantity']}")
                st.session_state["_reset_keep_fields"] = True
                st.rerun()
        elif withdraw_clicked:
            try:
                result = db.withdraw_item(desc, qty, manager_name=manager_name)
            except PermissionError:
                st.error("Close inventory.xlsx in Excel and try again.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Withdraw failed: {exc}")
            else:
                st.session_state["keep_status"] = (
                    "success", f"Withdrawn: {result['description']} → qty {result['quantity']} remaining")
                st.session_state["_reset_keep_fields"] = True
                st.rerun()


PAGES = {
    "gatekeeper": page_gatekeeper,
    "dashboard": page_dashboard,
    "admin_login": page_admin_login,
    "admin_panel": page_admin_panel,
    "enroll": page_enroll,
    "check": page_check,
    "item_detail": page_item_detail,
    "keep": page_keep,
}


def main():
    _init_session_state()
    PAGES[st.session_state.page]()


if __name__ == "__main__":
    main()
