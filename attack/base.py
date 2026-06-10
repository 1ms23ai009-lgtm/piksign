from typing import List

import einops
import torch
import torch.nn as nn
from tqdm import tqdm

from config_schema import MainConfig
from utils import log_metrics


class AttackFramework:
    """
    Base class for adversarial attack methods, three subclasses are only different in how to use gradient to update delta
    Core of the attack is to compute loss and apply transformation to the image, which is implemented in this class.
    """

    def __init__(
        self,
        cfg: MainConfig,
        ensemble_loss: nn.Module,
        source_crop: callable,
        target_crop: callable,
        change_iters: List[int],
    ):
        self.cfg = cfg
        self.ensemble_loss = ensemble_loss
        self.source_crop = source_crop
        self.target_crop = target_crop
        self.change_iters = change_iters

    @classmethod
    def create(cls, attack_type: str, **kwargs):
        """Factory method to create the appropriate attack instance"""
        attack_classes = {
            "fgsm": FGSMAttack,
            "mifgsm": MIFGSMAttack,
            "pgd": PGDAttack,
            "pgd_multi_pass": PGDAttackMutiPass,
        }

        if attack_type not in attack_classes:
            raise ValueError(f"Unknown attack type: {attack_type}")

        return attack_classes[attack_type](**kwargs)

    def attack(
        self,
        img_index: int,
        image_org: torch.Tensor,
        image_tgt: torch.Tensor,
        target_num: int = 1,
    ):
        """
        Perform attack to generate adversarial examples.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def initialize_delta(self, image_org: torch.Tensor):
        """Initialize the perturbation"""
        return torch.zeros_like(image_org, requires_grad=True)

    def compute_loss_and_metrics(
        self,
        adv_image: torch.Tensor,
        image_tgt: torch.Tensor,
        delta: torch.Tensor,
        log_global_sim: bool = False,
        target_num: int = 1,
        current_iter: int = 0,
    ):
        """Compute loss and metrics based on model configuration"""
        batch_size = adv_image.shape[0]

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # For global similarity (optional)
        if log_global_sim:
            with torch.no_grad():
                global_sim = self.ensemble_loss(adv_image, image_tgt)
            metrics["global_similarity"] = (
                global_sim.item() / batch_size
            )  # compute average similarity
        else:
            global_sim = -1

        # if multiple targets are not provided, repeat the target for each image
        if target_num > 1 and image_tgt.size(0) == adv_image.size(0):
            image_tgt = einops.repeat(image_tgt, "b c h w -> (b n) c h w", n=target_num)

        crop_idx = (
            0
            if current_iter < self.change_iters[0]
            else 1 if current_iter < self.change_iters[1] else 2
        )
        target_cropped = self.target_crop(image_tgt)
        source_cropped = self.source_crop[crop_idx](adv_image)
        local_sim = self.ensemble_loss(
            source_cropped, target_cropped, target_num=target_num
        )

        loss = local_sim
        metrics["local_similarity"] = local_sim / batch_size

        return loss, metrics, global_sim

    def finalize_image(self, image_org: torch.Tensor, delta: torch.Tensor):
        """Create the final adversarial image"""
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image, 0.0, 1.0)
        return adv_image


class FGSMAttack(AttackFramework):
    """Fast Gradient Sign Method attack"""

    def attack(
        self,
        img_index: int,
        image_org: torch.Tensor,
        image_tgt: torch.Tensor,
        target_num: int = 1,
        log_wandb: bool = True,
        log_interval: int = 25,
        disable_tqdm: bool = False,
    ):
        delta = self.initialize_delta(image_org)

        pbar = tqdm(
            range(self.cfg.optim.steps),
            desc=f"FGSM Attack progress",
            disable=disable_tqdm,
        )
        current_iter = 0
        for epoch in pbar:
            adv_image = image_org + delta
            current_iter += 1
            loss, metrics, _ = self.compute_loss_and_metrics(
                adv_image,
                image_tgt,
                delta,
                target_num=target_num,
                current_iter=current_iter,
            )

            if epoch % log_interval == 0:
                log_metrics(pbar, metrics, img_index, epoch, log_wandb=log_wandb)

            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

            delta.data = torch.clamp(
                delta + self.cfg.optim.alpha * torch.sign(grad),
                min=-self.cfg.optim.epsilon / 255,
                max=self.cfg.optim.epsilon / 255,
            )

        adv_image = self.finalize_image(image_org, delta)

        # Log final perturbation metrics
        final_metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, final_metrics, img_index, log_wandb=log_wandb)

        return adv_image


class MIFGSMAttack(AttackFramework):
    """Momentum Iterative Fast Gradient Sign Method attack"""

    def attack(
        self,
        img_index: int,
        image_org: torch.Tensor,
        image_tgt: torch.Tensor,
        target_num: int = 1,
        log_wandb: bool = True,
        log_interval: int = 25,
        disable_tqdm: bool = False,
    ):
        delta = self.initialize_delta(image_org)
        momentum = torch.zeros_like(image_org, requires_grad=False)

        pbar = tqdm(
            range(self.cfg.optim.steps),
            desc=f"MI-FGSM Attack progress",
            disable=disable_tqdm,
        )

        # Main optimization loop
        current_iter = 0
        for epoch in pbar:
            adv_image = image_org + delta
            current_iter += 1
            loss, metrics, _ = self.compute_loss_and_metrics(
                adv_image,
                image_tgt,
                delta,
                target_num=target_num,
                current_iter=current_iter,
            )

            if epoch % log_interval == 0:
                log_metrics(pbar, metrics, img_index, epoch, log_wandb=log_wandb)

            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

            momentum = momentum * self.cfg.optim.get("momentum_decay", 0.9) + grad
            delta.data = torch.clamp(
                delta + self.cfg.optim.alpha * torch.sign(momentum),
                min=-self.cfg.optim.epsilon / 255,
                max=self.cfg.optim.epsilon / 255,
            )

        adv_image = self.finalize_image(image_org, delta)

        final_metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }
        log_metrics(pbar, final_metrics, img_index, log_wandb=log_wandb)

        return adv_image


class PGDAttack(AttackFramework):
    """Projected Gradient Descent attack with configurable optimizer"""

    def attack(
        self,
        img_index: int,
        image_org: torch.Tensor,
        image_tgt: torch.Tensor,
        target_num: int = 1,
        amp: bool = True,
        log_wandb: bool = True,
        log_interval: int = 25,
        disable_tqdm: bool = False,
    ):
        delta = self.initialize_delta(image_org)
        optimizer = self.create_optimizer(delta)

        # Setup for mixed precision training
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp)
        autocast = torch.amp.autocast(device_type="cuda", enabled=amp)

        pbar = tqdm(
            range(self.cfg.optim.steps),
            desc=f"PGD Attack progress",
            disable=disable_tqdm,
        )

        current_iter = 0
        for epoch in pbar:
            optimizer.zero_grad()
            current_iter += 1

            with autocast:
                adv_image = image_org + delta
                loss, metrics, _ = self.compute_loss_and_metrics(
                    adv_image,
                    image_tgt,
                    delta,
                    target_num=target_num,
                    current_iter=current_iter,
                )
                loss = -loss

            if epoch % log_interval == 0:
                log_metrics(pbar, metrics, img_index, epoch, log_wandb=log_wandb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            delta.data = torch.clamp(
                delta,
                min=-self.cfg.optim.epsilon / 255,
                max=self.cfg.optim.epsilon / 255,
            )

        adv_image = self.finalize_image(image_org, delta)

        final_metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        log_metrics(pbar, final_metrics, img_index, log_wandb=log_wandb)

        return adv_image

    def create_optimizer(self, delta):
        """Create optimizer based on configuration"""
        optimizer_type = self.cfg.optim.get("optimizer", "adam")

        if optimizer_type.lower() == "adam":
            momentum = self.cfg.optim.get("momentum", 0.9)
            return torch.optim.Adam(
                [delta], lr=self.cfg.optim.alpha, betas=(momentum, 0.999)
            )
        elif optimizer_type.lower() == "sgd":
            momentum = self.cfg.optim.get("momentum", 0.9)
            return torch.optim.SGD([delta], lr=self.cfg.optim.alpha, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")


class PGDAttackMutiPass(AttackFramework):
    """Projected Gradient Descent attack with multiple forward-backward passes.
    Instead of replicating both source and target images (which would cause gradient explosion),
    this implementation performs separate forward-backward passes for each target and
    accumulates gradients before updating parameters once."""

    def attack(
        self,
        img_index: int,
        image_org: torch.Tensor,
        image_tgt: torch.Tensor,
        target_num: int = 1,
        amp: bool = True,
        log_wandb: bool = True,
        log_interval: int = 25,
        disable_tqdm: bool = False,
        multi_pass_num: int = 5,
    ):
        delta = self.initialize_delta(image_org)
        optimizer = self.create_optimizer(delta)

        # Setup for mixed precision training
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp)
        autocast = torch.amp.autocast(device_type="cuda", enabled=amp)

        pbar = tqdm(
            range(self.cfg.optim.steps),
            desc=f"PGD Multi-Pass Attack progress",
            disable=disable_tqdm,
        )

        # Prepare target images if target_num > 1
        # We'll process them one by one rather than replicating the source
        target_images = image_tgt
        if target_num > 1 and image_tgt.size(0) == image_org.size(0):
            # If we have a single target per source but need multiple targets,
            # repeat the target image along the batch dimension
            target_images = einops.repeat(
                image_tgt, "b c h w -> (b n) c h w", n=target_num
            )

        current_iter = 0
        for epoch in pbar:
            optimizer.zero_grad()
            current_iter += 1
            total_loss = 0
            combined_metrics = None

            # Process each target separately
            for t_idx in range(multi_pass_num):
                # Select the appropriate target for this pass
                current_target = target_images

                with autocast:
                    adv_image = image_org + delta
                    # For metrics, we pass target_num=1 since we're handling one target at a time
                    loss, metrics, _ = self.compute_loss_and_metrics(
                        adv_image,
                        current_target,
                        delta,
                        target_num=target_num,
                        current_iter=current_iter,
                    )
                    # Negate the loss as we want to maximize similarity
                    loss = -loss / multi_pass_num
                    total_loss -= loss.item() / multi_pass_num

                # Accumulate gradients without updating yet
                scaler.scale(loss).backward()

                # Aggregate metrics (first pass initializes, subsequent passes update)
                if combined_metrics is None:
                    combined_metrics = metrics.copy()
                else:
                    for key in metrics:
                        if key not in [
                            "max_delta",
                            "mean_delta",
                        ]:  # Skip delta metrics as they're the same
                            combined_metrics[key] += metrics[key]

            # Normalize metrics that were summed across targets
            if target_num > 1 and combined_metrics is not None:
                for key in combined_metrics:
                    if key not in ["max_delta", "mean_delta"]:
                        combined_metrics[key] /= target_num

            combined_metrics["local_similarity"] = total_loss

            if epoch % log_interval == 0:
                log_metrics(
                    pbar, combined_metrics, img_index, epoch, log_wandb=log_wandb
                )

            # Now update the parameters after all gradients are accumulated
            scaler.step(optimizer)
            scaler.update()

            # Project delta to the valid range
            delta.data = torch.clamp(
                delta,
                min=-self.cfg.optim.epsilon / 255,
                max=self.cfg.optim.epsilon / 255,
            )

        adv_image = self.finalize_image(image_org, delta)

        final_metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        log_metrics(pbar, final_metrics, img_index, log_wandb=log_wandb)

        return adv_image

    def create_optimizer(self, delta):
        """Create optimizer based on configuration"""
        optimizer_type = self.cfg.optim.get("optimizer", "adam")
        momentum = self.cfg.optim.get("momentum", 0.9)
        if optimizer_type.lower() == "adam":
            return torch.optim.Adam([delta], lr=self.cfg.optim.alpha, betas=(momentum, 0.999))
        elif optimizer_type.lower() == "sgd":
            return torch.optim.SGD([delta], lr=self.cfg.optim.alpha, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
