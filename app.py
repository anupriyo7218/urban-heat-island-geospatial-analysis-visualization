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

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    def load_tif(url):
        try:
            r = requests.get(url)
            img = Image.open(BytesIO(r.content))
            arr = np.array(img).astype(float)
            if arr.ndim > 2:
                arr = arr[:, :, 0]
            return arr[::5, ::5]
        except:
            return np.zeros((200, 200))

    B4 = "https://drive.google.com/uc?id=1-HBxOXTgztez8l3SyoT8BDL_spI-OpNy"
    B5 = "https://drive.google.com/uc?id=1Wh9InqB4Kzv3zzcHVd2TMsbRdlnU4710"
    B10 = "https://drive.google.com/uc?id=1r0ga3m4jWg0BVA2APWPt76Wauxh8bHwE"

    return load_tif(B4), load_tif(B5), load_tif(B10)

red, nir, thermal = load_data()

# -------- CLEAN --------
red = np.where(red == 0, np.nan, red)
nir = np.where(nir == 0, np.nan, nir)
thermal = np.where(thermal == 0, np.nan, thermal)

# -------- NDVI --------
ndvi = (nir - red) / (nir + red + 1e-10)
ndvi = np.clip(ndvi, -1, 1)

# -------- TEMP --------
temperature = thermal * 0.00341802 + 149.0
temperature_c = temperature - 273.15
temperature_c[(temperature_c < -50) | (temperature_c > 80)] = np.nan

# -------- RESCALE TEMP --------
def rescale(arr):
    v = arr[~np.isnan(arr)]
    if len(v) == 0: return arr
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-10)) * 25 + 20

temp_disp = rescale(temperature_c)

# -------- NORMALIZE --------
def norm(arr):
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10)

uhi = norm(temperature_c) - norm(ndvi)
uhi = (uhi - np.nanmean(uhi)) / (np.nanstd(uhi) + 1e-10)
uhi = np.clip(uhi, -2, 2)

# -------- IMAGE --------
def to_img(arr, cmap):
    c = plt.get_cmap(cmap)
    n = norm(arr)
    rgba = c(n)
    rgba[..., 3] = np.where(np.isnan(arr), 0, 1)
    return (rgba * 255).astype(np.uint8)

# -------- LEGEND --------
def legend(title, vmin, vmax):
    st.markdown(f"### 🎨 {title} | Min: {round(vmin,2)} | Max: {round(vmax,2)}")

# ===============================
# 🔥 COMPARISON MODE
# ===============================
if mode == "Comparison":

    st.subheader("📊 NDVI vs Temperature Comparison")
    st.info("📊 Comparison is fixed: NDVI vs Temperature (used to analyze Urban Heat Island effect)")

    c1, c2 = st.columns(2)
    c1.image(to_img(ndvi, "RdYlGn"), caption="NDVI")
    c2.image(to_img(temp_disp, "inferno"), caption="Temperature (°C)")

    nd = ndvi[~np.isnan(ndvi)]
    tp = temp_disp[~np.isnan(temp_disp)]
    n = min(len(nd), len(tp))
    nd, tp = nd[:n], tp[:n]

    st.subheader("📊 Comparison Statistics")

    col1, col2 = st.columns(2)
    col1.metric("NDVI Mean", round(np.mean(nd), 2))
    col2.metric("Temp Mean (°C)", round(np.mean(tp), 2))

    if n > 0:
        corr = np.corrcoef(nd, tp)[0,1]
        st.metric("Correlation", round(corr, 2))

        if corr < -0.5:
            st.success("🌿 Strong inverse relationship: Vegetation reduces temperature significantly.")
        elif corr < -0.2:
            st.info("🌱 Moderate inverse relationship: Vegetation helps reduce temperature.")
        else:
            st.warning("⚠️ Weak correlation observed. Urban temperature is influenced by multiple factors such as built-up areas, materials, and coastal effects, not just vegetation.")

    fig, ax = plt.subplots()
    ax.scatter(nd, tp, s=1)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

# ===============================
# 🔥 SINGLE LAYER
# ===============================
else:

    m = folium.Map(location=[19.07,72.87], zoom_start=10)

    if layer == "NDVI":
        data, img = ndvi, to_img(ndvi,"RdYlGn")
        legend("NDVI Scale", -1, 1)

    elif layer == "Temperature":
        data, img = temp_disp, to_img(temp_disp,"inferno")
        legend("Temperature (°C)", 20, 45)

    else:
        data, img = uhi, to_img(uhi,"coolwarm")
        legend("UHI Index", -2, 2)

    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[18.8,72.6],[19.3,73.2]],
        opacity=opacity
    ).add_to(m)

    st_folium(m, width=900, height=600)

    # -------- SUMMARY --------
    st.subheader("📊 Summary Statistics")

    v = data[~np.isnan(data)]

    c1, c2, c3 = st.columns(3)
    if len(v)>0:
        c1.metric("Mean", round(np.mean(v),2))
        c2.metric("Min", round(np.min(v),2))
        c3.metric("Max", round(np.max(v),2))

    # NDVI vegetation
    if layer=="NDVI" and len(v)>0:
        veg = v[v>0.3]
        st.info(f"🌱 Vegetation Coverage: {(len(veg)/len(v))*100:.2f}%")

    # UHI hotspot
    if layer=="UHI" and len(v)>0:
        hot = np.percentile(v,95)
        st.warning(f"🔥 UHI Hotspot Threshold: {round(hot,2)}")

    # -------- HIST --------
    st.subheader("📊 Data Distribution")
    fig, ax = plt.subplots()
    if len(v)>0:
        ax.hist(v, bins=50)
        ax.set_xlabel("Temperature (°C)" if layer=="Temperature" else "Value")
        ax.set_ylabel("Pixel Count")
        st.pyplot(fig)
    else:
        st.warning("No valid data")
