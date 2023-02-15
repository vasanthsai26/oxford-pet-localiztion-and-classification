import torch
import random
import os
import io
import config
import warnings
import matplotlib

import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A

from albumentations.pytorch import ToTensorV2 
from PIL import Image
from locnet import LocNet

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

TRANSFORMS = A.Compose(
    [
        A.Resize(256,256),
        A.CenterCrop(width=config.IMG_SIZE, height=config.IMG_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD, max_pixel_value=255,),
        ToTensorV2(),
    ])
# TRANSFORMS = transforms.Compose([
#                     transforms.Resize(256),
#                     transforms.CenterCrop(config.IMG_SIZE),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         config.IMAGENET_MEAN,
#                         config.IMAGENET_STD)])

def transform_image(image_bytes):
    """
    returns a tranformed pytorch image tensor
    """
    image = Image.open(io.BytesIO(image_bytes))
    target_image = TRANSFORMS(image=np.array(image))
    target_image = target_image["image"]
    return target_image.unsqueeze(0)

# Helper Functions
def seed_everything(TORCH_SEED: int) -> None:
    """
    Sets the manual SEED  
    """
    random.seed(TORCH_SEED)  
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)


def get_device() -> str:
    """
    Returns the default device available
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> LocNet:
    """
    return a trained LocNet model
    """
    seed_everything(config.SEED)
    model = LocNet(
        num_layers      = config.RESNET_LAYERS,
        image_channels  = config.IMAGE_CHANNELS,
        num_classes     = config.NUM_CLASSES,
        bb_points       = config.BOUNDING_BOX_POINTS,
        pretrained      = True).to(get_device())
    model.load_state_dict(
        torch.load(
            config.TRAINED_MODEL_DIR,
            map_location=torch.device('cpu')
            ))
    model.eval()
    return model

def denormalize(images, 
            means=config.IMAGENET_MEAN, 
            stds=config.IMAGENET_STD,
            device=get_device()):
    """
    denormalize the image with imagenet stats
    """
    means = torch.tensor(means).reshape(1, 3, 1, 1).to(device)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1).to(device)
    return (images * stds + means)

def get_predictions(model,target_image):
    """
    returns the model class label and bbox points
    """
    with torch.inference_mode():
        y_class_pred,y_bbox_preds = model(target_image.to(get_device()))

    image = denormalize(target_image).squeeze(axis=0).permute(1,2,0).cpu()
    y_label = torch.argmax(y_class_pred,dim=1).item()
    y_bbox_pascal  = [point.cpu().numpy().astype(np.uint8) for point in y_bbox_preds[0]]

    return image,y_label,y_bbox_pascal

def save_prediction(image,y_bbox_pascal):
    """
    saves the image with prediction and return the path
    only saves maximum of ten output images 
    """
    xmin   = y_bbox_pascal[0]
    ymin   = y_bbox_pascal[1]
    width  = (y_bbox_pascal[2]-y_bbox_pascal[0])
    height = (y_bbox_pascal[3]-y_bbox_pascal[1])

    image_name = f"output-{random.randint(0,9)}.png"
    image_dir = os.path.join(config.OUTPUT_IMAGE_DIR,image_name)

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(image)
    bbox = patches.Rectangle(
        (xmin,ymin),
        width,
        height,
        linewidth=2,
        edgecolor='g',
        facecolor='none')
    ax.add_patch(bbox)
    ax.set_axis_off()
    print()
    fig.savefig(image_dir)
    fig.show()
    return image_name









     

 



