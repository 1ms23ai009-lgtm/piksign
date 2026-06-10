from .base import BaseFeatureExtractor
from torchvision import transforms
from transformers import Dinov2Model
from typing import Optional, List

MODEL_DICT = {
    "dino_small": "facebook/dinov2-small",
    "dino_base": "facebook/dinov2-base",
    "dino_large": "facebook/dinov2-large",
}


class Dinov2FeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(Dinov2FeatureExtractor, self).__init__()
        self.model = Dinov2Model.from_pretrained(MODEL_DICT[model_name])
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def forward(self, x, return_dict=False):
        x = dict(pixel_values=self.normalizer(x))
        outputs = self.model(**x)
        if return_dict:
            return outputs
        else:
            return outputs.pooler_output
