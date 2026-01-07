# app/streamlit_app.py
# Streamlit App — Production-ready Inference (Model + Processor)
# Updated: professional sidebar with model/processor selection & load buttons
# ==============================================================

import streamlit as st
import pandas as pd
import pickle
import joblib
import os
from PIL import Image, ImageDraw, ImageOps
import glob
import numpy as np
from datetime import datetime
from typing import Optional, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart  



# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="Zomato Restaurant Classifier",
    layout="wide",
    page_icon="🍽️"
)

# -----------------------------------------------------------
# Heading Section (Centered)
# -----------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding-top: 10px;">
        <h1 style="margin-bottom: 0;">Zomato Restaurant Classifier — 🧠🍴</h1>
        <p style="font-size: 18px; color: #666;">
            Auto-loads or choose processor & model, applies preprocessing, then predicts.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# Resolve Image Path Dynamically
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # app/
project_root = os.path.dirname(current_dir)               # project root
assets_path = os.path.join(project_root, "assets")        # assets/
image_path = os.path.join(assets_path, "promo_zomato.png")# banner image

# -----------------------------------------------------------
# Image Display with Adjustable Width, Height, Padding, Rounded Border Inside Image
# -----------------------------------------------------------
if os.path.exists(image_path):
    # --- Image Settings Expander (Hidden by Default) ---
    with st.sidebar.expander("Image Settings", expanded=False):
        custom_width = st.slider("Width (px)", 100, 1200, 600)
        custom_height = st.slider("Height (px)", 100, 1000, 400)
        padding_top = st.slider("Top Padding (px)", 0, 100, 20)
        padding_bottom = st.slider("Bottom Padding (px)", 0, 100, 20)
        padding_left = st.slider("Left Padding (px)", 0, 100, 50)
        padding_right = st.slider("Right Padding (px)", 0, 100, 50)
        border_width = st.slider("Border Width (px)", 0, 20, 2)
        border_color = st.color_picker("Border Color", "#333333")
        corner_radius = st.slider("Corner Radius (px)", 0, 100, 20)

    # --- Load and Resize Image ---
    banner_img = Image.open(image_path).convert("RGBA")
    resized_img = banner_img.resize((custom_width, custom_height))

    # --- Create Rounded Mask ---
    mask = Image.new("L", resized_img.size, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle(
        [(0, 0), resized_img.size],
        radius=corner_radius,
        fill=255
    )

    # --- Apply mask to image (rounded corners) ---
    rounded_img = ImageOps.fit(resized_img, resized_img.size)
    rounded_img.putalpha(mask)

    # --- Create Canvas with Padding ---
    canvas_width = custom_width + padding_left + padding_right
    canvas_height = custom_height + padding_top + padding_bottom
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))

    # --- Paste Rounded Image on Canvas ---
    paste_x = padding_left
    paste_y = padding_top
    canvas.paste(rounded_img, (paste_x, paste_y), rounded_img)

    # --- Draw Border Inside Image ---
    if border_width > 0:
        draw = ImageDraw.Draw(canvas)
        border_rect = [
            (paste_x + border_width//2, paste_y + border_width//2),
            (paste_x + custom_width - border_width//2 - 1, paste_y + custom_height - border_width//2 - 1)
        ]
        draw.rounded_rectangle(
            border_rect,
            radius=corner_radius,
            outline=border_color,
            width=border_width
        )

    # --- Display Final Image ---
    st.image(canvas.convert("RGB"), use_container_width=False)

else:
    st.error(f"⚠️ Image not found at: {image_path}")


# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------
def get_files(path_pattern: str) -> List[str]:
    files = glob.glob(path_pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def get_latest(pattern: str) -> Optional[str]:
    files = get_files(pattern)
    return files[0] if files else None

def pretty_name(p: str) -> str:
    return os.path.basename(p)

# --------------------------------------------------------------
# Sidebar: load dataset (for dropdowns), choose processor & model
# --------------------------------------------------------------
st.sidebar.header("Configuration")

# ---- show dataset loaded (for dropdown options) ----
DATA_PATH = get_latest(os.path.join(PREPROCESSED_DIR, "zomato_preprocessed*.csv"))
if DATA_PATH:
    st.sidebar.success(f"📄 Loaded data: {pretty_name(DATA_PATH)}")
else:
    st.sidebar.warning("⚠ No preprocessed CSV found (used only for UI options).")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

df_org = None
if DATA_PATH:
    try:
        df_org = load_data(DATA_PATH)
    except Exception as e:
        st.sidebar.warning(f"Could not load preprocessed CSV: {e}")
        df_org = None

# ---- list processors and models ----
processor_candidates = get_files(os.path.join(MODELS_DIR, "processor_*.pkl"))
model_candidates = [p for p in get_files(os.path.join(MODELS_DIR, "*.pkl")) if "processor_" not in os.path.basename(p).lower()]

# Provide friendly "None found" messages while still letting user continue
proc_options = ["-- Select processor --"] + [pretty_name(p) for p in processor_candidates]
model_options = ["-- Select model --"] + [pretty_name(p) for p in model_candidates]

# Sidebar controls
auto_select_latest = st.sidebar.checkbox("Auto-select latest processor & model", value=True)

if auto_select_latest:
    selected_processor_name = pretty_name(processor_candidates[0]) if processor_candidates else "-- Select processor --"
    selected_model_name = pretty_name(model_candidates[0]) if model_candidates else "-- Select model --"
else:
    selected_processor_name = st.sidebar.selectbox("Processor (choose)", proc_options, index=0)
    selected_model_name = st.sidebar.selectbox("Model (choose)", model_options, index=0)

# Buttons for loading
st.sidebar.markdown("---")
col_load1, col_load2 = st.sidebar.columns([1,1])
with col_load1:
    load_btn = st.button("Load Selected Model ✅")
with col_load2:
    reload_btn = st.button("Reload Latest 🔁")

# Sidebar status placeholders
processor_status = st.sidebar.empty()
model_status = st.sidebar.empty()

# --------------------------------------------------------------
# Caching loaders (resource caching)
# --------------------------------------------------------------
@st.cache_resource
def load_processor(path: str):
    return joblib.load(path)

@st.cache_resource
def load_model(path: str):
    # Try joblib then pickle
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# --------------------------------------------------------------
# Manage session state for loaded processor/model
# --------------------------------------------------------------
if "processor_path" not in st.session_state:
    st.session_state.processor_path = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "model" not in st.session_state:
    st.session_state.model = None

# Helper: resolve names -> full paths
def name_to_path(name: str, candidates: List[str]):
    if not name or name.startswith("--"):
        return None
    for p in candidates:
        if os.path.basename(p) == name:
            return p
    return None

# If user chose auto-select, prefill selections
if auto_select_latest:
    if processor_candidates:
        selected_processor_name = pretty_name(processor_candidates[0])
    else:
        selected_processor_name = "-- Select processor --"
    if model_candidates:
        selected_model_name = pretty_name(model_candidates[0])
    else:
        selected_model_name = "-- Select model --"

# Respond to Load / Reload actions
if reload_btn:
    # force reload latest
    proc_path = processor_candidates[0] if processor_candidates else None
    mdl_path = model_candidates[0] if model_candidates else None
    if proc_path:
        try:
            st.session_state.processor = load_processor(proc_path)
            st.session_state.processor_path = proc_path
            processor_status.success(f"🛠️ Loaded processor: {pretty_name(proc_path)}")
        except Exception as e:
            processor_status.error(f"Failed to load processor: {e}")
    else:
        processor_status.warning("No processor file found.")

    if mdl_path:
        try:
            st.session_state.model = load_model(mdl_path)
            st.session_state.model_path = mdl_path
            model_status.success(f"🤖 Loaded model: {pretty_name(mdl_path)}")
        except Exception as e:
            model_status.error(f"Failed to load model: {e}")
    else:
        model_status.warning("No model file found.")

if load_btn:
    # load selected by name
    proc_path = name_to_path(selected_processor_name, processor_candidates)
    mdl_path = name_to_path(selected_model_name, model_candidates)

    # If user didn't explicitly choose, fall back to latest
    if proc_path is None and processor_candidates:
        proc_path = processor_candidates[0]
    if mdl_path is None and model_candidates:
        mdl_path = model_candidates[0]

    if proc_path:
        try:
            st.session_state.processor = load_processor(proc_path)
            st.session_state.processor_path = proc_path
            processor_status.success(f"🛠️ Loaded processor: {pretty_name(proc_path)}")
        except Exception as e:
            processor_status.error(f"Failed to load processor: {e}")
    else:
        processor_status.warning("No processor selected / available.")

    if mdl_path:
        try:
            st.session_state.model = load_model(mdl_path)
            st.session_state.model_path = mdl_path
            model_status.success(f"🤖 Loaded model: {pretty_name(mdl_path)}")
        except Exception as e:
            model_status.error(f"Failed to load model: {e}")
    else:
        model_status.warning("No model selected / available.")

# If session has loaded model/processor show them
if st.session_state.processor_path:
    processor_status.info(f"Processor: {pretty_name(st.session_state.processor_path)}")
else:
    processor_status.info("Processor: (not loaded)")

if st.session_state.model_path:
    model_status.info(f"Model: {pretty_name(st.session_state.model_path)}")
else:
    model_status.info("Model: (not loaded)")


st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    💡 **Quick Guide to Load and Use the Model**  

    1️⃣ **Select Your Files:**  
    Choose the processor and model files from the sidebar. These files are needed for the app to process data and make predictions.  

    2️⃣ **Load the Model:**  
    Click **`Load Selected Model`** to load the selected processor and model into the app.  

    3️⃣ **Reload Latest (Optional):**  
    If you want to use the newest saved files automatically, click **`Reload Latest`**.  

    ✅ Once loaded, the **`Predict`** button will become active, and you can start making predictions.
    """
)

# --------------------------------------------------------------
# Expected input columns (model input) — يجب أن تعكس الأعمدة التي دربت عليها processor
# If your processor expects different columns, update AVAILABLE_COLUMNS accordingly.
# --------------------------------------------------------------
AVAILABLE_COLUMNS = [
    'online_order','book_table','phone','approx_cost','menu_item',
    'listed_type','listed_city','rest_type',
    'votesRange','phone_lines','approx_cost_range','location_city','city_match',
    'Total_North_Indian','Total_South_Indian','Total_East_Indian',
    'Total_West_Indian','Total_International','Total_Asian',
    'Total_Grill_BBQ_Bar','Total_Fast_Food','Total_Beverages_Desserts',
    'Total_Healthy_Fusion','Total_Bakery',
    'Rest_SweetOrBakery','Rest_Drink_Oriented_Establishments',
    'Rest_Specialty_Shops','Rest_Dining_Establishments',
    'Rest_Takeaway_and_Delivery'
]

# --------------------------------------------------------------
# Helper: build dropdown options from dataset if available
# --------------------------------------------------------------
def safe_unique_list(df: Optional[pd.DataFrame], col: str):
    if df is None or col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())

listed_city_ops = ["Select"] + safe_unique_list(df_org, "listed_city")
listed_type_ops = ["Select"] + safe_unique_list(df_org, "listed_type")
rest_type_ops = ["Select"] + safe_unique_list(df_org, "rest_type")

# --------------------------------------------------------------
# UI — Prediction form (main area)
# --------------------------------------------------------------
st.header("🔮 Predict Restaurant Quality")
col1, col2, col3, col4 = st.columns(4)

with col1:
    listed_city = st.selectbox("City 🌆", listed_city_ops)
    approx_cost = st.number_input("Approx Cost 💸", 0.0, 20000.0, 200.0, step=10.0)
    Total_North_Indian = st.number_input("North Indian 🍛", 0, 2000, 0)
    Total_South_Indian = st.number_input("South Indian 🥥", 0, 2000, 0)
    Total_East_Indian = st.number_input("East Indian 🍲", 0, 2000, 0)
    Total_West_Indian = st.number_input("West Indian 🍛", 0, 2000, 0)
    Total_International = st.number_input("International 🌍", 0, 2000, 0)

with col2:
    listed_type = st.selectbox("Listed Type 🏷️", listed_type_ops)
    Total_Asian = st.number_input("Asian 🍜", 0, 2000, 0)
    Total_Grill_BBQ_Bar = st.number_input("Grill/BBQ/Bar 🍖", 0, 2000, 0)
    Total_Fast_Food = st.number_input("Fast Food 🍔", 0, 2000, 0)
    Total_Beverages_Desserts = st.number_input("Desserts 🍰", 0, 2000, 0)
    Total_Healthy_Fusion = st.number_input("Healthy/Fusion 🥗", 0, 2000, 0)
    Total_Bakery = st.number_input("Bakery 🥐", 0, 2000, 0)

with col3:
    online_order = st.radio("Online Order 📲", [0,1], horizontal=True)
    book_table = st.radio("Book Table 🪑", [0,1], horizontal=True)
    phone = st.radio("Phone ☎️", [0,1], horizontal=True)
    menu_item = st.radio("Menu Item 📋", [0,1], horizontal=True)
    rest_type = st.selectbox("Restaurant Type 🍽️", rest_type_ops)

with col4:
    Rest_SweetOrBakery = st.radio("Sweet/Bakery 🍰", [0,1], horizontal=True)
    Rest_Drink_Oriented_Establishments = st.radio("Drink-Oriented 🍹", [0,1], horizontal=True)
    Rest_Specialty_Shops = st.radio("Specialty Shops 🛍️", [0,1], horizontal=True)
    Rest_Dining_Establishments = st.radio("Dining 🍽️", [0,1], horizontal=True)
    Rest_Takeaway_and_Delivery = st.radio("Delivery 🛵", [0,1], horizontal=True)

# ------------------------------------------------------
# Build user input DataFrame
# ------------------------------------------------------
user_dict = {
    "online_order": online_order,
    "book_table": book_table,
    "phone": phone,
    "approx_cost": approx_cost,
    "menu_item": menu_item,
    "listed_type": None if listed_type=="Select" else listed_type,
    "listed_city": None if listed_city=="Select" else listed_city,
    "rest_type": None if rest_type=="Select" else rest_type,
    # missing columns filled as NaN for user input
    "votesRange": np.nan,
    "phone_lines": np.nan,
    "approx_cost_range": np.nan,
    "location_city": None,
    "city_match": None,
    "Total_North_Indian": Total_North_Indian,
    "Total_South_Indian": Total_South_Indian,
    "Total_East_Indian": Total_East_Indian,
    "Total_West_Indian": Total_West_Indian,
    "Total_International": Total_International,
    "Total_Asian": Total_Asian,
    "Total_Grill_BBQ_Bar": Total_Grill_BBQ_Bar,
    "Total_Fast_Food": Total_Fast_Food,
    "Total_Beverages_Desserts": Total_Beverages_Desserts,
    "Total_Healthy_Fusion": Total_Healthy_Fusion,
    "Total_Bakery": Total_Bakery,
    "Rest_SweetOrBakery": Rest_SweetOrBakery,
    "Rest_Drink_Oriented_Establishments": Rest_Drink_Oriented_Establishments,
    "Rest_Specialty_Shops": Rest_Specialty_Shops,
    "Rest_Dining_Establishments": Rest_Dining_Establishments,
    "Rest_Takeaway_and_Delivery": Rest_Takeaway_and_Delivery
}

user_df = pd.DataFrame([user_dict])
st.subheader("📋 Your Raw Input")
st.dataframe(user_df.T, use_container_width=True)

# ------------------------------------------------------
# Ensure all required columns exist
# ------------------------------------------------------
for col in AVAILABLE_COLUMNS:
    if col not in user_df.columns:
        user_df[col] = np.nan
user_df = user_df[AVAILABLE_COLUMNS]

with st.expander("🔧 Prepared input (before processor)"):
    st.write(user_df.T)

# ------------------------------------------------------
# Prediction button (disabled until model+processor are loaded)
# ------------------------------------------------------
predict_col, explain_col = st.columns([1,2])
with predict_col:
    predict_btn = st.button("Predict 🔍", disabled=(st.session_state.model is None or st.session_state.processor is None))

with explain_col:
    st.markdown(
        """
        **Instructions:**  
        1. In the sidebar: select or reload the latest processor and model.  
        2. Click `Load Selected Model` to load the files.  
        3. After loading, the `Predict` button will be enabled.  
        """
    )

# ------------------------------------------------------
# Prediction action
# ------------------------------------------------------
if predict_btn:
    try:
        processor = st.session_state.processor
        model = st.session_state.model
        if processor is None or model is None:
            st.error("Processor or model not loaded. Please load them from the sidebar first.")
        else:
            processed = processor.transform(user_df)

            if isinstance(processed, np.ndarray):
                proc_df = pd.DataFrame(processed, index=user_df.index, columns=[f"f{i}" for i in range(processed.shape[1])])
            else:
                # try to coerce to DataFrame (if transformer returns pandas)
                try:
                    proc_df = pd.DataFrame(processed)
                except Exception:
                    # fallback: numpy -> df
                    proc_df = pd.DataFrame(np.asarray(processed))

            st.subheader("⚙️ Processed Features (Debug)")
            st.write(proc_df.head())

            # Prediction
            pred = None
            try:
                pred = model.predict(proc_df)[0]
            except Exception as e:
                # Some models expect 2D numeric arrays (not df) or different dtype
                try:
                    pred = model.predict(proc_df.values)[0]
                except Exception as e2:
                    raise e  # re-raise original so we can show proper traceback

            if pred == 1 or str(pred) == "1":
                st.success("🎉 Prediction: **GOOD Restaurant**")
                st.balloons()
            else:
                st.error("❌ Prediction: **NOT GOOD**")

            # show probabilities if available
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(proc_df)[0]
                    st.write("Probabilities:", proba)
            except Exception:
                pass

    except Exception as e:
        st.error("❌ Prediction Error — See details below")
        st.exception(e)

# --------------------------------------------------------------
# Debug / Info section
# --------------------------------------------------------------
with st.expander("🧪 Debug Info"):
    st.write("Processor loaded path:", st.session_state.processor_path)
    st.write("Model loaded path:", st.session_state.model_path)
    st.write("Timestamp:", datetime.now().isoformat())
    if df_org is not None:
        st.write("Dataset sample (first 5 rows):")
        st.dataframe(df_org.head())
    st.write("Required columns used for prediction:")
    st.write(AVAILABLE_COLUMNS)

# --------------------------------------------------------------
# Footer / Helpful checks
# --------------------------------------------------------------
# --------------------------------------------------------------
# Footer / Developer Info Section
# --------------------------------------------------------------
st.markdown("---")
st.subheader("👨‍💻 Developer Information")

st.markdown("""
**This application was fully developed and engineered by:**

### 🧑‍💻 **Abdallah Nabil Ragab**  
**Data Scientist | Machine Learning Engineer | Software Engineer**  
**M.Sc. in Business Information Systems**

If you have any suggestions, ideas, feature requests, or want to report issues,  
please feel free to send your feedback directly via email:

📩 **Email:** `abdallah.nabil.ragab94@gmail.com`  

I appreciate your thoughts and feedback that help improve this project.  
""")

# -------------------------------------------------------------------
# cd C:\Users\user\Downloads\project_structure\app   ----------------
# venv\Scripts\activate.bat                          ----------------
# streamlit run streamlit_app.py                     ----------------
# -------------------------------------------------------------------

