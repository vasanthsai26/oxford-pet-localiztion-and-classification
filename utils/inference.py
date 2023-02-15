import json
import config
import time

from helper import get_model,transform_image,get_predictions,save_prediction

### open and save the idx_to_label json
with open(config.IDX_LABEL_JSON_DIR,"r") as json_file:
    idx_to_label = json.load(json_file)

### Load the saved model   
locnet_model = get_model()

def get_labels(idx):
    return idx_to_label['class_label'][str(idx)],idx_to_label['species_label'][str(idx)]

def make_predictions(image_path):
    start = time.time()
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    

    target_image = transform_image(image_bytes=image_bytes)
    image,y_label,y_bbox_pascal = get_predictions(locnet_model,target_image)

    output_image = save_prediction(image,y_bbox_pascal)
    breed,species = get_labels(y_label)
    end = time.time()

    image_details = {
        "image_name"  : output_image,
        "species_name": species,
        "breed_name"  : breed,
        "pred_time"   : end - start,
        }
    return image_details


# with open(os.path.join(config.INPUT_IMAGE_DIR,'Abyssinian_204.jpg'), 'rb' ) as f:
#     image_bytes = f.read()
#     print(make_predictions(image_bytes=image_bytes))








