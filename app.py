import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import requests
from io import BytesIO

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
    opacity = 0.7  # default (used internally, not shown)

# -------- LOAD DATA FROM GOOGLE DRIVE --------
@st.cache(allow_output_mutation=True)
def load_data():

    def load_tif(url):
        response = requests.get(url)
        return rasterio.open(BytesIO(response.content))

    B4_URL = "https://drive.google.com/uc?id=1-HBxOXTgztez8l3SyoT8BDL_spI-OpNy"
    B5_URL = "https://drive.google.com/uc?id=1Wh9InqB4Kzv3zzcHVd2TMsbRdlnU4710"
    B10_URL = "https://drive.google.com/uc?id=1r0ga3m4jWg0BVA2APWPt76Wauxh8bHwE"

    with load_tif(B4_URL) as src:
        red = src.read(1)[::5, ::5]

    with load_tif(B5_URL) as src:
        nir = src.read(1)[::5, ::5]

    with load_tif(B10_URL) as src:
        thermal = src.read(1)[::5, ::5]

    return red, nir, thermal

red, nir, thermal = load_data()

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

# -------- RESCALE TEMPERATURE --------
def rescale_temp(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return arr

    lower = np.percentile(valid, 2)
    upper = np.percentile(valid, 98)

    clipped = np.clip(arr, lower, upper)
    norm = (clipped - lower) / (upper - lower + 1e-10)
    scaled = norm * (45 - 20) + 20

    return scaled

temperature_display = rescale_temp(temperature_c)

# -------- NORMALIZATION --------
def safe_norm(arr):
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

ndvi_norm = safe_norm(ndvi)
temp_norm = safe_norm(temperature_c)

# -------- UHI --------
uhi = temp_norm - ndvi_norm
mean = np.nanmean(uhi)
std = np.nanstd(uhi)
uhi_std = (uhi - mean) / (std + 1e-10)
uhi_std = np.clip(uhi_std, -2, 2)

# -------- IMAGE FUNCTION --------
def array_to_image(arr, cmap_name):
    cmap = cm.get_cmap(cmap_name)
    valid = ~np.isnan(arr)

    if np.any(valid):
        norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10)
    else:
        norm = np.zeros_like(arr)

    rgba = cmap(norm)
    rgba[..., 3] = np.where(np.isnan(arr), 0, 1)

    return (rgba * 255).astype(np.uint8)

# -------- LEGEND --------
def add_legend(title, vmin, vmax):
    st.markdown(f"### 🎨 {title} Scale | Min: {round(vmin,2)} | Max: {round(vmax,2)}")

# ===============================
# 🔥 COMPARISON MODE
# ===============================
if mode == "Comparison":
    st.subheader("📊 NDVI vs Temperature Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.image(array_to_image(ndvi, 'RdYlGn'), caption="NDVI (Vegetation)")

    with col2:
        st.image(array_to_image(temperature_display, 'inferno'), caption="Temperature (°C)")
        
    st.info("📊 Comparison is fixed: NDVI vs Temperature (used to analyze Urban Heat Island effect)")

    ndvi_valid = ndvi[~np.isnan(ndvi)]
    temp_valid = temperature_display[~np.isnan(temperature_display)]

    min_len = min(len(ndvi_valid), len(temp_valid))
    ndvi_valid = ndvi_valid[:min_len]
    temp_valid = temp_valid[:min_len]

    st.subheader("📊 Comparison Statistics")

    col1, col2 = st.columns(2)
    col1.metric("NDVI Mean", round(np.mean(ndvi_valid), 2))
    col2.metric("Temp Mean (°C)", round(np.mean(temp_valid), 2))

    if min_len > 0:
        corr = np.corrcoef(ndvi_valid, temp_valid)[0, 1]
        st.metric("NDVI vs Temperature Correlation", round(corr, 2))

        if corr < -0.5:
            st.success("🌿 Strong inverse relationship: Vegetation reduces temperature significantly.")
        elif corr < -0.2:
            st.info("🌱 Moderate inverse relationship: Vegetation helps reduce temperature.")
        else:
            st.warning("⚠️ Weak correlation observed. Urban temperature is influenced by multiple factors such as built-up areas, materials, and coastal effects, not just vegetation.")

    st.subheader("📉 NDVI vs Temperature Relationship")

    fig, ax = plt.subplots()
    ax.scatter(ndvi_valid, temp_valid, s=1)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

# ===============================
# 🔥 SINGLE LAYER MODE
# ===============================
else:
    m = folium.Map(location=[19.07, 72.87], zoom_start=10)

    if layer == "NDVI":
        data = ndvi
        img = array_to_image(ndvi, 'RdYlGn')
        add_legend("NDVI", -1, 1)

    elif layer == "Temperature":
        data = temperature_display
        img = array_to_image(temperature_display, 'inferno')
        add_legend("Temperature (°C)", 20, 45)

    else:
        data = uhi_std
        img = array_to_image(uhi_std, 'coolwarm')
        add_legend("UHI Index", -2, 2)

    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[18.8, 72.6], [19.3, 73.2]],
        opacity=opacity
    ).add_to(m)

    st_folium(m, width=900, height=600)

    st.subheader("📊 Summary Statistics")

    valid = data[~np.isnan(data)]

    col1, col2, col3 = st.columns(3)

    if len(valid) > 0:
        col1.metric("Mean", round(np.mean(valid), 2))
        col2.metric("Min", round(np.min(valid), 2))
        col3.metric("Max", round(np.max(valid), 2))
    else:
        st.warning("No valid data")

    if layer == "NDVI":
        veg = ndvi[~np.isnan(ndvi)]
        green = veg[veg > 0.3]
        if len(veg) > 0:
            st.info(f"🌱 Vegetation Coverage: {(len(green)/len(veg))*100:.2f}%")

    if layer == "UHI":
        hotspot = np.percentile(valid, 95)
        st.warning(f"🔥 UHI Hotspot Threshold: {round(hotspot,2)}")

    st.subheader("📊 Data Distribution")

    fig, ax = plt.subplots()

    if len(valid) > 0:
        ax.hist(valid, bins=50)
        ax.set_xlabel("Temperature (°C)" if layer == "Temperature" else "Value")
        ax.set_ylabel("Pixel Count")
        st.pyplot(fig)
    else:
        st.warning("No valid data available")
