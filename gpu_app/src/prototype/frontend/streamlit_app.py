import streamlit as st
import requests
import os
import leafmap.foliumap as leafmap
import geopandas as gpd
from pathlib import Path


# backend server URL
BACKEND_URL = "http://localhost:5000"

def geojson_download_button(path: Path, label: str):
    # Only works if results are ready after pipeline run
    if not st.session_state.get("results_ready", False):
        st.button(label, disabled=True)
        return

    if not path.exists():
        st.warning(f"{path.name} not found")
        return

    with open(path, "rb") as f:
        st.download_button(
            label=label,
            data=f,
            file_name=path.name,
            mime="application/geo+json"
        )


st.set_page_config(layout="wide")
st.title("Tree Segmentation Demo")

if "results_ready" not in st.session_state:
    st.session_state["results_ready"] = False


# Health Check
if st.button("Check Server Health"):
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            health = response.json()
            st.success(
                f"Server ready | GPU: {health['gpu']} | "
                f"Free Memory: {health['free_mem_mb']} MB"
            )
        else:
            st.error("Server responded with an error!")
    except Exception as e:
        st.error(f"Could not reach server: {e}")

st.divider()

# Run Segmentation
if st.button("Run Tree Segmentation"):
    try:
        with st.spinner("Running segmentation pipeline on GPU..."):
            response = requests.post(f"{BACKEND_URL}/process")
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Segmentation finished! Objects detected: {result['objects']}"
                )

                st.session_state["results_ready"] = True

                st.rerun()

            else:
                st.error(f"Pipeline error: {response.text}")
    except Exception as e:
        st.error(f"Could not reach server: {e}")



# Map Visualization 
st.subheader("Segmentation Results")


# Paths to GeoJSON results
DINO_GEOJSON = Path("/storage/soltau/data/prototype_results/dino_output/detections.geojson")
SAM_GEOJSON  = Path("/storage/soltau/data/prototype_results/sam_output/masks.geojson")

# Leafmap with initial setup on Lüneburg center
m = leafmap.Map(center=[53.25, 10.41], zoom=17)
m.add_basemap("SATELLITE")

# DINO GeoJSON Layer
if DINO_GEOJSON.exists():
    m.add_geojson(str(DINO_GEOJSON), layer_name="DINO Boxes", style={"color": "red", "weight": 2, "fillOpacity": 0})

# SAM GeoJSON Layer
if SAM_GEOJSON.exists():
    m.add_geojson(str(SAM_GEOJSON), layer_name="SAM Masks", style={"color": "green", "weight": 1, "fillColor": "green", "fillOpacity": 0.7})

# Map in Streamlit einbetten
m.to_streamlit(height=700)


# Download Buttons
st.subheader("Download Results")

col1, col2 = st.columns(2)

with col1:
    geojson_download_button(
        DINO_GEOJSON,
        "Download DINO Bounding Boxes (GeoJSON)"
    )

with col2:
    geojson_download_button(
        SAM_GEOJSON,
        "Download SAM Masks (GeoJSON)"
    )




