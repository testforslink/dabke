import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision
from segment_anything_hq import sam_model_registry, SamPredictor
import os
from segment_anything import SamPredictor
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

#from Grounding DINO
import grouninDINO.groundingdino.datasets.transforms as T
from grouninDINO.groundingdino.models import build_model
from grouninDINO.groundingdino.util.slconfig import SLConfig
from grouninDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()







def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def extract_bounding_box(image_np, text_prompt, model, device='cuda', box_threshold=0.3, text_threshold=0.25):
    # Convert the NumPy array to a PIL Image
# Convert the NumPy array to a PIL Image
# Convert the NumPy array to a PIL Image
    print('whem 11111111 xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    image_pil = Image.fromarray(image_np)
    # model accepts the image in form of pil the 
    # Transform the image
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('when 2222 xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    transformed_image, _ = transform(image_pil, None)

    # Move the image to the specified device
    print('when 3333333 xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    image = transformed_image.to(device).unsqueeze(0)
    print('when brurhrhhrrhhrrhrhhrrhhr  xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    
    print('when lmaowwwwwwwwwwwwwwww  xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # Convert text prompt to lowercase and strip
    text_prompt = text_prompt.lower().strip()
    if not text_prompt.endswith("."):
        text_prompt += "."

    with torch.no_grad():
        print('when teriiiimummyy11111  xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model.to(device)
        outputs = model(image, captions=[text_prompt])
        print('when teriiiimummyy 22222  xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('when 444444444 xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    logits = outputs["pred_logits"].to(device).sigmoid()[0]
    boxes = outputs["pred_boxes"].to(device)[0]
    print('when 55555555 xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    # Filter output based on box_threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    print('when 6666666666 xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # Get phrases from positive map
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(text_prompt)
    print('when 777777777777 xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    bounding_boxes = []
    print('when yyyyyyyyyyyy xxxxxxxxxxxxxxxxxxxxxxxxxxx')

    size = image_pil.size
  

    return bounding_boxes

def main2 (imageurl, text_prompt,respath,hq_token_only):
    print('start')
    image_bgr = cv2.imread(imageurl)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    device = "cuda"
    model_type = "vit_h"
    # image_pil = input_image["image"].convert("RGB")
    

    sam_checkpoint = "./pretrained_checkpoint/sam_hq_vit_h.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_predictor = SamPredictor(sam)

    mask_predictor.set_image(image_rgb)
    
    model_checkpoint = './groundingdino_swint_ogc.pth'
    Config_file = './grouninDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    print('loading model xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    grouninDINO_model = load_model(
    Config_file, model_checkpoint, device=device)
    coordnatebondingbox = extract_bounding_box(image_rgb, text_prompt, grouninDINO_model)
    box = np.array(coordnatebondingbox)    
    masks, scores, logits = mask_predictor.predict(
     box=box,
        hq_token_only=hq_token_only
    )

    # mask_annotator = sv.MaskAnnotator()
    # detections = sv.Detections.from_sam(sam_result=masks)
    # annotated_image = mask_annotator.annotate(scene=image_bgr.copy(),detections= detections)
    
    # newimage = Image.fromarray(annotated_image)
    
    
    # # example save path ./demo/output_imgs/which.jpg
    # newimage.save(respath)

    # masks = masks.squeeze(1).cpu().numpy()
    # scores = scores.squeeze(1).cpu().numpy()
    # box = box.cpu().numpy()
    input_point, input_label = None, None
    show_res_multi(masks, scores, input_point, input_label, box, respath, image_rgb)

    return respath
 

def Main():
    
    main2('demo/input_imgs/watch.jpg','subject','demo/output_imgs/which2',False)


if __name__ == '__main__':
        Main()
