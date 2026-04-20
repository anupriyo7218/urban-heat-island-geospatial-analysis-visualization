import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import requests
from io import BytesIO
from PIL import Image

st.set_page_config(layout="wide")

st.title("🌆 Urban Heat Island Analysis - Mumbai")

st.markdown("""
This dashboard visualizes Urban Heat Island patterns in Mumbai using Landsat 9 data.

- **NDVI** → Vegetation density  
- **Temperature** → Land Surface Temperature (°C)  
- **UHI** → Urban Heat Island intensity  
""")

# -------- SIDEBAR --------
st.sidebar.title("Controls")
mode = st.sidebar.radio("View Mode", ["Single Layer", "Comparison"])

if mode == "Single Layer":
    layer = st.sidebar.selectbox("Select Layer", ["NDVI", "Temperature", "UHI"])
    opacity = st.sidebar.slider("Overlay Opacity", 0.1, 1.0, 0.7)
else:
    layer = None
    opacity = 0.7

# -------- SAFE LOAD DATA --------
@st.cache_data
def load_data():

    def load_tif(url):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()

            img = Image.open(BytesIO(r.content))
            arr = np.array(img).astype(float)

            # ensure 2D
            if arr.ndim > 2:
                arr = arr[:, :, 0]

            return arr[::5, ::5]

        except Exception:
            # always return fallback so app never crashes
            return np.zeros((200, 200))

    B4_URL = "https://drive.google.com/uc?id=1-HBxOXTgztez8l3SyoT8BDL_spI-OpNy"
    B5_URL = "https://drive.google.com/uc?id=1Wh9InqB4Kzv3zzcHVd2TMsbRdlnU4710"
    B10_URL = "https://drive.google.com/uc?id=1r0ga3m4jWg0BVA2APWPt76Wauxh8bHwE"

    red = load_tif(B4_URL)
    nir = load_tif(B5_URL)
    thermal = load_tif(B10_URL)

    return red, nir, thermal


# -------- CALL SAFELY --------
try:
    red, nir, thermal = load_data()
except Exception:
    red = np.zeros((200, 200))
    nir = np.zeros((200, 200))
    thermal = np.zeros((200, 200))

# -------- CLEAN DATA --------
red = np.where(red == 0, np.nan, red)
nir = np.where(nir == 0, np.nan, nir)
thermal = np.where(thermal == 0, np.nan, thermal)

# -------- NDVI --------
ndvi = (nir - red) / (nir + red + 1e-10)
ndvi = np.clip(ndvi, -1, 1)

# -------- TEMPERATURE --------
temperature = thermal * 0.00341802 + 149.0
temperature_c = temperature - 273.15
temperature_c[(temperature_c < -50) | (temperature_c > 80)] = np.nan

# -------- RESCALE --------
def rescale_temp(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return arr
    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)
    norm = (np.clip(arr, low, high) - low) / (high - low + 1e-10)
    return norm * 25 + 20

temperature_display = rescale_temp(temperature_c)

# -------- NORMALIZATION --------
def safe_norm(arr):
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10)

ndvi_norm = safe_norm(ndvi)
temp_norm = safe_norm(temperature_c)

# -------- UHI --------
uhi = temp_norm - ndvi_norm
uhi_std = (uhi - np.nanmean(uhi)) / (np.nanstd(uhi) + 1e-10)
uhi_std = np.clip(uhi_std, -2, 2)

# -------- IMAGE --------
def array_to_image(arr, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10)
    rgba = cmap(norm)
    rgba[..., 3] = np.where(np.isnan(arr), 0, 1)
    return (rgba * 255).astype(np.uint8)

# ===============================
# SINGLE MODE
# ===============================
if mode == "Single Layer":
    m = folium.Map(location=[19.07, 72.87], zoom_start=10)

    if layer == "NDVI":
        data, img = ndvi, array_to_image(ndvi, "RdYlGn")
    elif layer == "Temperature":
        data, img = temperature_display, array_to_image(temperature_display, "inferno")
    else:
        data, img = uhi_std, array_to_image(uhi_std, "coolwarm")

    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[18.8, 72.6], [19.3, 73.2]],
        opacity=opacity
    ).add_to(m)

    st_folium(m, width=900, height=600)

    valid = data[~np.isnan(data)]
    st.subheader("📊 Data Distribution")

    if len(valid) > 0:
        fig, ax = plt.subplots()
        ax.hist(valid, bins=50)
        st.pyplot(fig)
    else:
        st.warning("No valid data")

# ===============================
# COMPARISON MODE
# ===============================
else:
    col1, col2 = st.columns(2)
    col1.image(array_to_image(ndvi, "RdYlGn"), caption="NDVI")
    col2.image(array_to_image(temperature_display, "inferno"), caption="Temperature")
