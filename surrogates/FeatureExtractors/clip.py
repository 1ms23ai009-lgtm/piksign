from typing import List, Optional

import open_clip
import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from .base import BaseFeatureExtractor

MODEL_DICT = {
    "clip_b16": "openai/clip-vit-base-patch16",
    "clip_b32": "openai/clip-vit-base-patch32",
    "clip_l14": "openai/clip-vit-large-patch14",
    "clip_laion_bigg14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "clip_laion_g14": "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
    "clip_laion_h14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "clip_laion_b32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "clip_laion_b16": "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
}


class ClipFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for CLIP models that are hosted on HuggingFace.
    """

    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(ClipFeatureExtractor, self).__init__()
        open_clip_model_name = [
            "clip_laion_convnext_large",
            "clip_laion_convnext_large_ft",
            "clip_laion_convnext_large_ft_soup",
            "clip_laion_b16",
        ]
        # init HF or OpenClip model based on
        if model_name in open_clip_model_name:
            self.model = OpenClipFeatureExtractor(model_name, mean, std)
        else:
            self.model = HFClipFeatureExtractor(model_name, mean, std)

    def forward(self, x, return_dict=False):
        return self.model(x, return_dict)


class HFClipFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for CLIP models that are hosted on HuggingFace.
    """

    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(HFClipFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained(MODEL_DICT[model_name])
        self.model = self.model.vision_model
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
        """
        if return_dict is True, return the dictionary of all hidden states.
        if return_dict is False, return the pooler output of the last layer.
        """
        x = self.normalizer(x)

        output_attentions = False
        output_hidden_states = True
        interpolate_pos_encoding = False
        return_pooler_output = True

        dict_output = self.model(
            pixel_values=x,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_pooler_output,
        )
        pooler_output = dict_output.pooler_output
        if return_dict:
            image_features = dict_output
        else:
            image_features = pooler_output

        return image_features


class OpenClipFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for CLIP models that are hosted on HuggingFace.
    """

    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(OpenClipFeatureExtractor, self).__init__()
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(MODEL_DICT[model_name])
        )
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

    def forward(self, x, return_hidden_states=False):
        assert (
            return_hidden_states is False
        ), "OpenClip does not support return_hidden_states"
        x = self.normalizer(x)
        image_features = self.model.encode_image(x)
        return image_features
