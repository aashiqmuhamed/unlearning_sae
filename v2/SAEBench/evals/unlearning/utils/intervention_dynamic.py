import torch
import einops

from torch import Tensor
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from contextlib import contextmanager
from functools import partial
from sae_lens import SAE

import numpy as np


def anthropic_clamp_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    sae: SAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0,
    activation_threshold: float = 0.05,
    random: bool = False,
) -> Float[Tensor, "batch seq d_model"]:
    """
    Enhanced version of Anthropic's feature clamping that adds dynamic filtering based on activation patterns.
    """
    if len(features_to_ablate) > 0:
        # Encode residual stream and zero out BOS token
        feature_activations = sae.encode(resid)
        feature_activations[:, 0, :] = 0.0
        
        # Get initial reconstruction and error term
        reconstruction = sae.decode(feature_activations)
        error = resid - reconstruction

        # Create mask for features that exceed activation threshold
        target_features = feature_activations[:, :, features_to_ablate]
        activation_mask = target_features > 0
        
        # Calculate which batches exceed the activation threshold
        batch_activation_rates = activation_mask.sum(dim=(1, 2)) / (
            activation_mask.shape[1] * activation_mask.shape[2]
        )
        active_batches = batch_activation_rates > activation_threshold
        
        # Create final mask combining feature activation and batch threshold
        final_mask = activation_mask & active_batches.unsqueeze(1).unsqueeze(2)
        
        # Apply clamping to selected features
        feature_activations[:, :, features_to_ablate] = torch.where(
            final_mask,
            torch.full_like(target_features, -multiplier),
            feature_activations[:, :, features_to_ablate]
        )

        # Reconstruct and add back error term
        modified_reconstruction = sae.decode(feature_activations)
        resid = modified_reconstruction + error

    return resid

# def anthropic_clamp_resid_SAE_features(
#     resid: Float[Tensor, "batch seq d_model"],
#     hook: HookPoint,
#     sae: SAE,
#     features_to_ablate: list[int],
#     multiplier: float = 1.0,
#     random: bool = False,
# ) -> Float[Tensor, "batch seq d_model"]:
#     """
#     Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
#     used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
#     This version clamps the feature activation to the value(s) specified in multiplier
#     """

#     if len(features_to_ablate) > 0:
#         with torch.no_grad():
#             # adjust feature activations with scaling (multiplier = 0 just ablates the feature)
#             feature_activations = sae.encode(resid)

#             feature_activations[:, 0, :] = 0.0  # We zero out the BOS token for all SAEs.
#             # We don't need to zero out padding tokens because we right pad, so they don't effect the model generation.

#             reconstruction = sae.decode(feature_activations)

#             error = resid - reconstruction

#             non_zero_features_BLD = feature_activations[:, :, features_to_ablate] > 0


#             if not random:
#                 if isinstance(multiplier, float) or isinstance(multiplier, int):
#                     feature_activations[:, :, features_to_ablate] = torch.where(
#                         non_zero_features_BLD,
#                         -multiplier,
#                         feature_activations[:, :, features_to_ablate],
#                     )
#                 else:
#                     raise NotImplementedError("Currently deprecated")
#                     feature_activations[:, :, features_to_ablate] = torch.where(
#                         non_zero_features_BLD,
#                         -multiplier.unsqueeze(dim=0).unsqueeze(dim=0),
#                         feature_activations[:, :, features_to_ablate],
#                     )

#             else:
#                 raise NotImplementedError("Currently deprecated")
#                 assert isinstance(multiplier, float) or isinstance(multiplier, int)

#                 next_features_to_ablate = [
#                     (f + 1) % feature_activations.shape[-1] for f in features_to_ablate
#                 ]
#                 feature_activations[:, :, next_features_to_ablate] = torch.where(
#                     feature_activations[:, :, features_to_ablate] > 0,
#                     -multiplier,
#                     feature_activations[:, :, next_features_to_ablate],
#                 )

#             modified_reconstruction = sae.decode(feature_activations)

#             resid = error + modified_reconstruction
#         return resid
