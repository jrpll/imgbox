from PIL import Image
import torch
from torchvision import transforms

class BackgroundRemover:
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, pil_image):
        x = self.transform(pil_image).unsqueeze(0).to("cuda")
        pred = self.model(x)[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(pil_image.size)
        out = pil_image.copy()
        out.putalpha(mask)
        return out
