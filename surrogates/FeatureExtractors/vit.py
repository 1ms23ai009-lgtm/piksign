import torch
from transformers import ViTModel, ViTForImageClassification
from .base import BaseFeatureExtractor
from torchvision import transforms
from typing import List, Optional

MODEL_DICT = {
    "vit_b16": "google/vit-base-patch16-224-in21k",
    "vit_b32": "google/vit-base-patch32-224-in21k",
    "vit_h14": "google/vit-huge-patch14-224-in21k",
}


class VisionTransformerFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(VisionTransformerFeatureExtractor, self).__init__()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.model = ViTModel.from_pretrained(MODEL_DICT[model_name])
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean,
                    std,
                ),  # CLIP imgs mean and std.
            ]
        )

    def forward(self, x, return_dict=False):
        pixel_values = self.normalizer(x)
        vision_outputs = self.model(pixel_values, output_hidden_states=True)
        if return_dict:
            return vision_outputs
        else:
            return vision_outputs.logits
