import os


## DIR_CONFIGS
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
INPUT_IMAGE_DIR   = os.path.join(WORKING_DIR,"static/input-images")
OUTPUT_IMAGE_DIR  = os.path.join(WORKING_DIR,"static/output-images")
TRAINED_MODEL_DIR = os.path.join(WORKING_DIR,"model/trained-model.pth")
IDX_LABEL_JSON_DIR= os.path.join(WORKING_DIR,"static/idx_to_label.json")

## DATA CONFIGS
IMG_SIZE = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 37
BOUNDING_BOX_POINTS = 4
RESNET_LAYERS = 50
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225] 