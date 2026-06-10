from typing import Dict, List

import einops
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from surrogates.FeatureExtractors.base import BaseFeatureExtractor


def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """Compute cosine similarity between two tensors."""
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    # For normalized vectors, dot product equals cosine similarity
    # No need to divide by feature size as normalization handles this
    return (a * b).sum(dim=-1)


class EnsAlignmentLoss(nn.Module):
    """Integrated class that combines feature extraction and alignment loss calculation.

    This class simplifies the workflow by directly computing the alignment loss between
    source and target images in a single pass, without needing separate extraction
    and ground truth setting steps.
    """

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsAlignmentLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(
        self, source_img: Tensor, target_img: Tensor, target_num: int = 1
    ) -> Tensor:
        """Calculate alignment loss between source and target images.

        Args:
            source_img: Source image tensor to align
            target_img: Target image tensor to align with

        Returns:
            Tensor: Alignment loss value, computed from cosine similarity
        """
        loss = 0

        # Extract target features with no_grad
        target_features = []
        with torch.no_grad():
            for model in self.extractors:
                target_features.append(model(target_img).to(target_img.device))

        # Extract source features
        for index, model in enumerate(self.extractors):
            source_feature = model(source_img).squeeze()
            target_feature = target_features[index]
            loss += torch.sum(cosine_similarity(source_feature, target_feature))

        loss = loss / len(self.extractors)

        return loss

    def _get_normalized_features(self, model, x: Tensor) -> Tensor:
        """Extract features from input images by model. Used to compute cosine similarity."""
        features = model(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


class EnsMultiAlignmentLoss(EnsAlignmentLoss):
    """Integrated class that combines feature extraction and alignment loss calculation.

    This class simplifies the workflow by directly computing the alignment loss between
    source and target images in a single pass, without needing separate extraction
    and ground truth setting steps.
    """

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsMultiAlignmentLoss, self).__init__(extractors)

    def forward(
        self, source_img: Tensor, target_img: Tensor, target_num: int = 1
    ) -> Tensor:
        """Calculate alignment loss between source and multiple target images.

        Args:
            source_img: Source image tensor to align
            target_img: Target image tensor(s) to align with
            target_num: Number of target images per source image

        Returns:
            Tensor: Alignment loss value, computed from cosine similarity
        """
        loss = 0

        # Extract target features with no_grad
        target_features = []
        weights = []
        with torch.no_grad():
            target_features = []
            for model in self.extractors:
                target_features.append(
                    model(target_img, return_dict=False).to(target_img.device)
                )

        source_features = []
        for model in self.extractors:
            source_features.append(model(source_img, return_dict=False).squeeze())

        # Compute alignment loss
        for index, _ in enumerate(self.extractors):
            source_feature = source_features[index]  # [batch_size, feature_dim]
            target_feature = target_features[
                index
            ]  # [batch_size * target_num, feature_dim] or [batch_size, feature_dim]

            if target_num > 1:
                source_feature = einops.repeat(
                    source_feature, "b d -> (b n) d", n=target_num
                )
                similarity = cosine_similarity(source_feature, target_feature)
                loss += torch.sum(similarity) / target_num
            else:
                loss += torch.sum(cosine_similarity(source_feature, target_feature))

        loss = loss / len(self.extractors)

        return loss


class EnsCropMultiAlignmentLoss(EnsAlignmentLoss):
    """Integrated class that combines feature extraction and alignment loss calculation.

    This class simplifies the workflow by directly computing the alignment loss between
    source and target images in a single pass, without needing separate extraction
    and ground truth setting steps.
    """

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsCropMultiAlignmentLoss, self).__init__(extractors)

    def forward(
        self, source_img: Tensor, target_img: Tensor, target_num: int = 1
    ) -> Tensor:
        """Calculate alignment loss between source and multiple target images.

        Args:
            source_img: Source image tensor to align
            target_img: Target image tensor(s) to align with
            target_num: Number of target images per source image

        Returns:
            Tensor: Alignment loss value, computed from cosine similarity
        """
        loss = 0

        # Extract target features with no_grad
        target_features = []
        weights = []
        with torch.no_grad():
            target_features = []
            for model in self.extractors:
                if target_num > 1:
                    target_img = einops.repeat(
                        target_img, "b c h w -> (b n) c h w", n=target_num
                    )
                target_features.append(
                    model(target_img, return_dict=False).to(target_img.device)
                )

        source_features = []
        for model in self.extractors:
            if target_num > 1:
                source_img = einops.repeat(
                    source_img, "b c h w -> (b n) c h w", n=target_num
                )
            source_features.append(model(source_img, return_dict=False).squeeze())

        # Compute alignment loss
        for index, _ in enumerate(self.extractors):
            source_feature = source_features[index]  # [batch_size, feature_dim]
            target_feature = target_features[
                index
            ]  # [batch_size * target_num, feature_dim] or [batch_size, feature_dim]

            similarity = cosine_similarity(source_feature, target_feature)
            loss += torch.sum(similarity) / target_num

        loss = loss / len(self.extractors)

        return loss


def sequence_average_cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """First average over the sequence length, then compute cosine similarity."""
    # Normalize feature vectors
    a = torch.mean(a, dim=1)
    b = torch.mean(b, dim=1)
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute token-wise cosine similarity
    cosine_similarity = (a * b).sum(dim=-1)

    return cosine_similarity


def sequence_token_cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """Compute cosine similarity between two sequences of features."""
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    cosine_similarity = (a * b).sum(dim=-1)

    # average over the sequence length
    cosine_similarity = torch.mean(cosine_similarity, dim=1)
    return cosine_similarity


class EnsMultiHiddenStateLoss(EnsAlignmentLoss):
    """Integrated class that combines feature extraction and alignment loss calculation.
    Align both the hidden states and the pooler output of the source and target images.
    """

    def __init__(self, extractors: List[BaseFeatureExtractor], align_idx: List[int]):
        super(EnsMultiHiddenStateLoss, self).__init__(extractors)
        self.align_idx = align_idx

    def forward(
        self, source_img: Tensor, target_img: Tensor, target_num: int = 1
    ) -> Tensor:
        # Extract target features with no_grad
        target_features = []
        with torch.no_grad():
            for model in self.extractors:
                model_hidden_states = []
                model_output = model(target_img, return_dict=True)

                for idx in self.align_idx:
                    model_hidden_states.append(model_output.hidden_states[idx])

                if hasattr(model_output, "pooler_output"):
                    model_hidden_states.append(model_output.pooler_output.unsqueeze(1))

                target_features.append(model_hidden_states)

        # Extract source features
        source_features = []
        for model in self.extractors:
            source_hidden_states = []
            model_output = model(source_img, return_dict=True)

            for idx in self.align_idx:
                source_hidden_states.append(model_output.hidden_states[idx])

            if hasattr(model_output, "pooler_output"):
                source_hidden_states.append(model_output.pooler_output.unsqueeze(1))

            source_features.append(source_hidden_states)

        # Compute alignment loss
        total_loss = 0
        for src_feature_list, tgt_feature_list in zip(source_features, target_features):
            # Compute similarity for each feature type (hidden states + pooler)
            layer_similarities = []

            for src_embedding, tgt_embedding in zip(src_feature_list, tgt_feature_list):

                if target_num > 1:
                    src_embedding = torch.cat([src_embedding] * target_num, dim=0)

                similarity = sequence_token_cosine_similarity(
                    src_embedding, tgt_embedding
                )
                layer_similarities.append(similarity)

            # Average similarities across layers and sum across batch
            model_loss = torch.sum(
                torch.mean(torch.stack(layer_similarities, dim=1), dim=0)
            )
            total_loss += model_loss

        # Average across all models
        final_loss = total_loss / len(self.extractors)

        return final_loss


class EnsWeightedMultiAlignmentLoss(EnsAlignmentLoss):
    """
    Computes alignment loss between a source image and multiple target images,
    applying a weight (beta) to the loss contributions from the retrieved
    (non-primary) target images.

    Loss = Loss(src, primary_tgt) + beta * Sum(Loss(src, retrieved_tgt_i))
    """

    def __init__(self, extractors: List[BaseFeatureExtractor], beta: float = 1.0):
        super(EnsWeightedMultiAlignmentLoss, self).__init__(extractors)
        if not 0.0 <= beta <= 1.0:
            print(f"Warning: Beta value {beta} is outside the typical [0, 1] range.")
        self.beta = beta

    def forward(
        self, source_img: Tensor, target_img: Tensor, target_num: int = 1
    ) -> Tensor:
        """Calculate weighted alignment loss between source and multiple target images.

        Args:
            source_img: Source image tensor to align [batch_size, C, H, W]
            target_img: Target image tensor(s) to align with
                        [batch_size * target_num, C, H, W]
            target_num: Number of target images per source image

        Returns:
            Tensor: Weighted alignment loss value.
        """
        total_loss = 0.0
        batch_size = source_img.shape[0]
        # Extract target features with no_grad
        target_features_per_model = []
        with torch.no_grad():
            for model in self.extractors:
                # Shape: [batch_size * target_num, feature_dim]
                features = model(target_img, return_dict=False).to(target_img.device)
                target_features_per_model.append(features)

        # Extract source features
        source_features_per_model = []
        for model in self.extractors:
            # Shape: [batch_size, feature_dim]
            features = model(source_img, return_dict=False).squeeze()
            source_features_per_model.append(features)

        # Compute weighted alignment loss
        for index, _ in enumerate(self.extractors):  # Iterate through models
            source_feature = source_features_per_model[index]  # [b, d]
            target_feature = target_features_per_model[index]  # [b_n, d]

            if target_num > 1:
                n_retrieved = target_num - 1
                # Reshape target features: (b n) d -> b n d
                target_feature_reshaped = rearrange(
                    target_feature, "(b n) d -> b n d", b=batch_size
                )

                # Primary target features: [b, d]
                primary_target_feature = target_feature_reshaped[:, 0, :]

                # Retrieved target features: b n d -> b n_retrieved d
                retrieved_target_features = target_feature_reshaped[:, 1:, :]
                # Flatten retrieved targets: b n_retrieved d -> (b n_retrieved) d
                retrieved_target_features_flat = rearrange(
                    retrieved_target_features, "b n d -> (b n) d"
                )

                # Expand source features for retrieved targets: b d -> (b n_retrieved) d
                source_feature_expanded_retrieved = repeat(
                    source_feature, "b d -> (b n) d", n=n_retrieved
                )

                primary_similarity = cosine_similarity(
                    source_feature, primary_target_feature
                )

                # Retrieved similarities: [(b n_retrieved)]
                retrieved_similarity_flat = cosine_similarity(
                    source_feature_expanded_retrieved, retrieved_target_features_flat
                )

                # Sum over batch and retrieved dimensions
                model_primary_loss = primary_similarity.sum(dim=0)
                model_retrieved_loss = retrieved_similarity_flat.sum(dim=0)

                model_loss = model_primary_loss + self.beta * model_retrieved_loss
                total_loss += model_loss

            else:  # target_num == 1
                # Only primary target exists
                similarity = cosine_similarity(source_feature, target_feature)
                total_loss += similarity.sum()  # Sum over batch

        # Average loss across all models
        final_loss = total_loss / len(self.extractors) / target_num

        return final_loss


class EnsWeightedMultiCLSAlignmentLoss(EnsAlignmentLoss):
    """
    Computes weighted alignment loss between the CLS tokens of source and
    multiple target images at specified hidden layers.

    Loss = Loss_CLS(src, primary_tgt) + beta * Sum(Loss_CLS(src, retrieved_tgt_i))
    where Loss_CLS is computed only on the CLS token embeddings at layers in align_idx.
    """

    def __init__(
        self,
        extractors: List[BaseFeatureExtractor],
        align_idx: List[int],
        beta: float = 1.0,
    ):
        super(EnsWeightedMultiCLSAlignmentLoss, self).__init__(extractors)
        self.align_idx = align_idx
        if not 0.0 <= beta <= 1.0:
            print(f"Warning: Beta value {beta} is outside the typical [0, 1] range.")
        self.beta = beta
        print(
            f"Using EnsWeightedMultiCLSAlignmentLoss with align_idx={align_idx}, beta={beta}"
        )

    def forward(
        self, source_img: Tensor, target_img: Tensor, target_num: int = 1
    ) -> Tensor:
        """Calculate weighted CLS token alignment loss.

        Args:
            source_img: Source image tensor [batch_size, C, H, W]
            target_img: Target image tensor(s) [batch_size * target_num, C, H, W]
            target_num: Number of target images per source image

        Returns:
            Tensor: Weighted CLS alignment loss value.
        """
        batch_size = source_img.shape[0]

        # --- Extract Target Features (CLS tokens only) ---
        target_features_per_model = (
            []
        )  # List[List[Tensor]] - Outer: models, Inner: layers
        with torch.no_grad():
            for model in self.extractors:
                model_cls_features = []
                # Expects hidden_states and potentially pooler_output
                model_output = model(target_img, return_dict=True)

                # Extract CLS token from specified hidden states
                for idx in self.align_idx:
                    # hidden_state shape: [b*n, seq_len, dim]
                    cls_token_feature = model_output.hidden_states[idx][
                        :, 0, :
                    ]  # Select CLS token
                    model_cls_features.append(cls_token_feature.to(target_img.device))

                # Optionally add CLS token from pooler output if it exists and is different
                # Assuming pooler_output is [b*n, dim] or equivalent after projection
                if (
                    hasattr(model_output, "pooler_output")
                    and model_output.pooler_output is not None
                ):
                    # Pooler output might already be CLS-like, shape [b*n, dim]
                    pooler_cls = model_output.pooler_output.to(target_img.device)
                    # Avoid adding duplicate if last hidden state CLS is effectively the pooler
                    if (
                        not torch.allclose(
                            model_cls_features[-1], pooler_cls, atol=1e-6
                        )
                        or not self.align_idx
                        or self.align_idx[-1] != len(model_output.hidden_states) - 1
                    ):
                        model_cls_features.append(pooler_cls)

                target_features_per_model.append(model_cls_features)

        # --- Extract Source Features (CLS tokens only) ---
        source_features_per_model = (
            []
        )  # List[List[Tensor]] - Outer: models, Inner: layers
        for model in self.extractors:
            model_cls_features = []
            model_output = model(source_img, return_dict=True)

            for idx in self.align_idx:
                # hidden_state shape: [b, seq_len, dim]
                cls_token_feature = model_output.hidden_states[idx][
                    :, 0, :
                ]  # Select CLS token
                model_cls_features.append(cls_token_feature)

            if (
                hasattr(model_output, "pooler_output")
                and model_output.pooler_output is not None
            ):
                pooler_cls = model_output.pooler_output
                if (
                    not torch.allclose(model_cls_features[-1], pooler_cls, atol=1e-6)
                    or not self.align_idx
                    or self.align_idx[-1] != len(model_output.hidden_states) - 1
                ):
                    model_cls_features.append(pooler_cls)

            source_features_per_model.append(model_cls_features)

        # --- Compute Weighted Alignment Loss ---
        model_losses = []
        for model_idx, _ in enumerate(self.extractors):
            source_cls_layers = source_features_per_model[model_idx]  # List[[b, d]]
            target_cls_layers = target_features_per_model[model_idx]  # List[[b*n, d]]

            layer_losses = []
            # Iterate through aligned layers (CLS tokens)
            for src_cls_feat, tgt_cls_feat in zip(source_cls_layers, target_cls_layers):
                # src_cls_feat: [b, d], tgt_cls_feat: [b*n, d]

                if target_num > 1:
                    n_retrieved = target_num - 1
                    # Reshape target: (b n) d -> b n d
                    tgt_cls_feat_reshaped = rearrange(
                        tgt_cls_feat, "(b n) d -> b n d", b=batch_size
                    )

                    primary_tgt_cls = tgt_cls_feat_reshaped[:, 0, :]  # [b, d]
                    retrieved_tgt_cls = tgt_cls_feat_reshaped[:, 1:, :]  # [b, n-1, d]
                    # Flatten retrieved: b (n-1) d -> (b*(n-1)) d
                    retrieved_tgt_cls_flat = rearrange(
                        retrieved_tgt_cls, "b n d -> (b n) d", n=n_retrieved
                    )

                    # Expand source for retrieved: b d -> (b*(n-1)) d
                    src_cls_feat_expanded_retrieved = repeat(
                        src_cls_feat, "b d -> (b n) d", n=n_retrieved
                    )

                    # Similarities
                    primary_sim = cosine_similarity(
                        src_cls_feat, primary_tgt_cls
                    )  # [b]
                    retrieved_sim_flat = cosine_similarity(
                        src_cls_feat_expanded_retrieved, retrieved_tgt_cls_flat
                    )  # [b*(n-1)]

                    # Sum losses for this layer
                    layer_loss = (
                        primary_sim.sum() + self.beta * retrieved_sim_flat.sum()
                    )
                    layer_losses.append(layer_loss)

                else:  # target_num == 1
                    similarity = cosine_similarity(src_cls_feat, tgt_cls_feat)  # [b]
                    layer_losses.append(similarity.sum())  # Sum over batch

            # Average loss across layers for this model
            model_loss = torch.mean(
                torch.stack(layer_losses)
            )  # Average loss across the aligned layers
            model_losses.append(model_loss)

        # Average loss across all models
        final_loss = torch.mean(torch.stack(model_losses)) / batch_size

        return final_loss
