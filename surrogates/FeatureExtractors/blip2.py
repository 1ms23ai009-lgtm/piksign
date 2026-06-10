from typing import List
import torch
from transformers import (
    Blip2VisionModel,
    Blip2VisionConfig,
    Blip2Processor,
    Blip2Model,
    BlipImageProcessor,
)
from torchvision import transforms
from .base import BaseFeatureExtractor
from typing import Optional

MODEL_DICT = {
    "blip2_2.7b": "Salesforce/blip2-opt-2.7b",
    "blip2_6.7b": "Salesforce/blip2-opt-6.7b",
    "blip2_flan": "Salesforce/blip2-flan-t5-xl",
    "blip2_coco": "Salesforce/blip2-opt-6.7b-coco",
}


class BlipFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(BlipFeatureExtractor, self).__init__()
        self.model = Blip2Model.from_pretrained(MODEL_DICT[model_name])
        self.model.language_model = None
        self.model.qformer = None
        self.model.language_projection = None
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def forward(self, x, return_dict=False):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.get_image_features(**inputs)
        if return_dict:
            return outputs
        else:
            pooler_output = outputs.pooler_output
            image_features = pooler_output
            return image_features
