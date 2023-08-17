from transformers import CLIPVisionModel,AutoProcessor
from torchvision import transforms
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
raw_img_dir = r'../data/conan/150_boy/8b40ab2cbe2e1b5ea09b44969ccb1e4da438eac8.jpg'
raw_pil_img = Image.open(raw_img_dir)
# shape = 1(batch), 3(channel), h, w
clip_processored = processor(images=raw_pil_img, return_tensors="pt").pixel_values
img_features = img_encoder(clip_processored).pooler_output
print(f'img_features : {img_features}')