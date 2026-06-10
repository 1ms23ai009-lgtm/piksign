from .base import BaseFeatureExtractor
from transformers import BlipModel, BlipForQuestionAnswering, BlipForImageTextRetrieval, BlipVisionModel, BlipForConditionalGeneration
from torchvision import transforms
from typing import Optional, List

MODEL_DICT = {
    "blip1_base": "Salesforce/blip-image-captioning-base",
    "blip1_large": "Salesforce/blip-image-captioning-large",
    "blip1_vqa": "Salesforce/blip-vqa-base",
}


class Blip1FeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(Blip1FeatureExtractor, self).__init__()
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_DICT[model_name])
        self.model = self.model.vision_model
        #self.model.text_projection = None
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

    def forward(self, x):
        
        return_dict = self.model.config.use_return_dict
        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states
        inputs = dict(pixel_values=self.normalizer(x))

        vision_outputs = self.model(
            pixel_values=inputs["pixel_values"],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        pooler_output = vision_outputs.pooler_output
        image_features = pooler_output
        return image_features
