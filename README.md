TREE SEGMENTATION PROTOTYPE
CLONE THE REPOSITORY

To get started, clone the repository to your local machine:

git clone https://github.com/BentMildner/AITreeCounting.git
cd gpu_app



ABOUT THE PROJECT

This project is a prototype that addresses the challenge of detecting and segmenting trees automatically using AI models GroundingDINO and SamGeo2.
The implementation is realized through the LangSam class, which combines text-based object detection and segmentation. We use the official PyPI package segment-geospatial.

The goal is to provide an estimation of green coverage for the city of Lüneburg based on aerial imagery.
The designated outputs are:

Bounding boxes from DINO as a GeoJSON file

Masks from SAM as GeoTIFF files

DOCKER

The project can also be run using Docker for easier deployment on GPU-enabled servers.

A Dockerfile is provided for the backend environment

A docker-compose.yml allows you to build and run the service with GPU support

Volumes are mounted to persist input and output data

Example Docker command:

docker-compose up --build


This starts the backend Flask server and ensures all dependencies, including PyTorch and CUDA, are available inside the container.

HOW TO RUN THE PROJECT

Start the backend

python src/prototype/backend/app.py


The backend exposes REST endpoints (/health and /process) to manage the pipeline.

Start the frontend

streamlit run src/prototype/frontend/streamlit_app.py


The Streamlit app visualizes segmentation results and allows downloading GeoJSON masks.

Notes:

The project is designed to run on a GPU-enabled server. CPU startup is not tested.

Ensure input/output directories are correctly defined in config.py.

To reproduce results, use the campus .tif input image as provided in the example.

THE ARCHITECTURE

Backend Application: Flask REST API

Frontend Application: Streamlit

AI Models: GroundingDINO + SamGeo2

THE DATA

Input data is based on open-source data for Lower Saxony, focusing on the Leuphana Campus in Lüneburg.

Product used: Digital Orthophotos (DOP)

Resolution: ~20 cm per pixel

Source: LGN Open Geodata DOP

For demonstration, a 1024 x 1024 pixel tile covering the campus is used as input, ensuring a concise and reproducible example.

RESULTS

The prototype successfully detects most trees in the provided .tif.

Bounding boxes are created with DINO

Refined masks are produced by SAM

Results can be visualized on an interactive map via Streamlit + Leafmap

LIMITATIONS

False segmentation may occur in shadowed areas

Not all trees are detected in dense areas

Improvements could include:

Optimizing filtering functions

Tuning model hyperparameters

Using additional datasets

Training a custom smaller model for the campus
