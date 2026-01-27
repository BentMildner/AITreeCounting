# File for configuration of constannts and Hyperparameters

# Hardware
GPU_ID = "1"

# Paths
INPUT_PATH = "/storage/soltau/data/prototype_input/campus.tif"
DINO_OUTPUT_DIR = "/storage/soltau/data/prototype_results/dino_output"
SAM_OUTPUT_DIR = "/storage/soltau/data/prototype_results/sam_output"

# Model Parameters
MODEL_TYPE = "sam2-hiera-large"
PROMPT = "tree" 
BOX_THRESHOLD = 0.09 # almost every tree detected with this threshold
TEXT_THRESHOLD = 0.25

# Logger Setup
import logging
logger = logging.getLogger("LangSam")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)