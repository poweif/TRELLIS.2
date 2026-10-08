from typing import *
import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # RMBG-2.0 is gated on Hugging Face. Opt in to the ungated BiRefNet weights from the
        # model's author (MIT; not identical to RMBG-2.0) with TRELLIS2_UNGATED_MODELS=1.
        if model_name == "briaai/RMBG-2.0" and os.environ.get("TRELLIS2_UNGATED_MODELS") == "1":
            print(f"[TRELLIS2_UNGATED_MODELS] Loading ZhengPeng7/BiRefNet instead of {model_name}")
            model_name = "ZhengPeng7/BiRefNet"
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    