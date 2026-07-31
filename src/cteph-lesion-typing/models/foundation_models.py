"""VoCo foundation model wrapper for feature extraction."""

from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


# --- VoCo pretrained weights handling ---

VOCO_PRETRAINED_WEIGHTS_DIR = Path("./voco_pretrained_weights")


def _ssl_false():
    """Disable SSL verify for corporate proxy."""
    import httpx

    return httpx.Client(follow_redirects=True, verify=False)


class VoCoModelVariant(str, Enum):
    """VoCo model variants with associated parameters."""

    BASE = "base"
    LARGE = "large"
    HUGE = "huge"

    @property
    def feature_size(self) -> int:
        """Get the feature size for this model variant."""
        return {"base": 48, "large": 96, "huge": 192}[self.value]

    @property
    def weight_filename(self) -> str:
        """Get the weight filename for this model variant."""
        return f"VoComni_{self.value[0].upper()}.pt"

    @property
    def default_weight_path(self) -> Path:
        """Get the default weight path for this model variant."""
        return VOCO_PRETRAINED_WEIGHTS_DIR / self.weight_filename


def _load_weights(model, model_dict):
    """
    Load pretrained weights into a model, handling potential size mismatches.

    Matches keys by name and size. If a key exists in the checkpoint and has
    the same size as the model's parameter, it will be loaded; otherwise,
    the model's original parameter is retained.
    """
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    new_state_dict = {
        k: state_dict[k]
        if (k in state_dict.keys())
        and (state_dict[k].size() == current_model_dict[k].size())
        else current_model_dict[k]
        for k in current_model_dict.keys()
    }
    model.load_state_dict(new_state_dict, strict=True)
    return model


def _download_voco_model(
    model_weight_name: str, cache_dir: Path = VOCO_PRETRAINED_WEIGHTS_DIR
) -> str:
    """Download VoCo weights from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import set_client_factory

    set_client_factory(_ssl_false)

    repo_id = "Luffy503/VoCo"
    print(f"Downloading {model_weight_name} from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_weight_name,
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f"Model downloaded successfully to: {file_path}")
    return file_path


def load_voco_model(
    variant: VoCoModelVariant | str = VoCoModelVariant.HUGE,
    pretrained_path: Path | str | None = None,
    in_channels: int = 1,
    out_channels: int = 21,
) -> SwinUNETR:
    """
    Load a VoCo pretrained SwinUNETR model.

    Args:
        variant: Model variant (base, large, or huge).
        pretrained_path: Path to pretrained weights. If None, downloads from HF.
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        Loaded SwinUNETR model with pretrained weights.
    """
    if isinstance(variant, str):
        variant = VoCoModelVariant(variant.lower())

    if pretrained_path is not None:
        weight_path = str(pretrained_path)
    else:
        weight_path = _download_voco_model(
            model_weight_name=variant.weight_filename,
            cache_dir=VOCO_PRETRAINED_WEIGHTS_DIR,
        )

    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=variant.feature_size,
        use_v2=True,
    )

    model_dict = torch.load(
        weight_path, map_location=torch.device("cpu"), weights_only=False
    )
    model = _load_weights(model, model_dict)
    return model


# --- Feature Extractor Wrapper ---


class VoCoFeatureExtractor(nn.Module):
    """
    Wrapper around VoCo (SwinUNETR) for feature extraction.

    Uses multi-scale features from the SwinViT encoder to produce
    discriminative representations.

    .. warning::
        Stages 3-4 of SwinViT produce nearly identical features for
        any input (cosine similarity ≈ 1.0).  Prefer stages 0-2.

    Args:
        in_channels: Number of input channels (1 for CT).
        variant: VoCoModelVariant enum (BASE, LARGE, HUGE).
        freeze_backbone: If True, freeze all backbone parameters.
        pool: If True, forward() returns adaptive-pooled features (B, C).
              If False, forward() returns the spatial feature map from stage 0.
        stages: Stages to extract when pool=True (default: (0, 1, 2)).
        pool_size: Adaptive pool target size (default: 2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        variant: VoCoModelVariant = VoCoModelVariant.HUGE,
        freeze_backbone: bool = True,
        pool: bool = True,
        stages: tuple[int, ...] = (0, 1, 2),
        pool_size: int = 2,
    ):
        super().__init__()
        self.pool = pool
        self.variant = variant
        self.stages = stages

        # Load pretrained SwinUNETR backbone
        self.backbone = load_voco_model(
            variant=variant,
            in_channels=in_channels,
            out_channels=21,
        )

        # Pooling layers
        self.adaptive_avg = nn.AdaptiveAvgPool3d(pool_size)
        self.adaptive_max = nn.AdaptiveMaxPool3d(pool_size)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input volume.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            If pool=True: (B, feature_dim) concatenated multi-scale features.
            If pool=False: (B, C, D, H, W) spatial feature map from stage 0.
        """
        # Get multi-stage feature maps from SwinViT encoder
        features_list = self.backbone.swinViT(x)

        if self.pool:
            parts = []
            for s in self.stages:
                feat = features_list[s]
                parts.append(self.adaptive_avg(feat).flatten(1))
                parts.append(self.adaptive_max(feat).flatten(1))
            return torch.cat(parts, dim=1)
        else:
            # Return highest resolution stage
            return features_list[0]


# --- Classifier Wrapper ---


class VoCoClassifier(nn.Module):
    """
    Classification model built on top of a VoCo (SwinUNETR) backbone.

    Uses **multi-scale feature aggregation** from the SwinViT encoder:
    each selected stage is independently pooled (adaptive avg + max) and
    the resulting vectors are concatenated before the classification head.

    Why not just use the deepest stage?
    -----------------------------------------------------------------
    SwinViT stages 3-4 produce nearly identical features for any input
    (cosine similarity ≈ 1.0 after global average pooling) because the
    spatial resolution is too low (4³ / 2³) and LayerNorm collapses the
    information.  Stages 0-2 retain discriminative spatial patterns.

    Stage channel multipliers (relative to ``feature_size``):
        stage 0 → ×1,  stage 1 → ×2,  stage 2 → ×4,
        stage 3 → ×8,  stage 4 → ×16

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (1 for CT).
        variant: VoCoModelVariant enum (BASE, LARGE, HUGE).
        pretrained_path: Optional path to pretrained weights.
        freeze_backbone: If True, backbone parameters are frozen at init.
        dropout: Dropout probability before the final linear layer.
        stages: Tuple of stage indices to extract features from (0-4).
                Default ``(0, 1, 2)`` uses the three most discriminative
                stages while ignoring the collapsed deeper ones.
        pool_size: Spatial size for adaptive pooling before flattening
                   (default 2 → each stage gives C×2³ = 8C features).
        head_hidden_dim: Hidden layer size; defaults to ``feature_dim // 2``.

    Example:
        >>> model = VoCoClassifier(num_classes=2, variant=VoCoModelVariant.HUGE)
        >>> out = model(x)  # (B, num_classes)
        >>> model.unfreeze_backbone()
    """

    # SwinViT stage channel multipliers
    _STAGE_MULTIPLIERS = (1, 2, 4, 8, 16)

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        variant: VoCoModelVariant = VoCoModelVariant.HUGE,
        pretrained_path: Path | str | None = None,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        stages: tuple[int, ...] = (0, 1, 2),
        pool_size: int = 2,
        head_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.variant = variant
        self.stages = stages
        self.pool_size = pool_size
        self._backbone_frozen = False

        # Load pretrained SwinUNETR backbone
        self.backbone = load_voco_model(
            variant=variant,
            pretrained_path=pretrained_path,
            in_channels=in_channels,
            out_channels=21,
        )

        # Adaptive pooling shared by all stages
        self.adaptive_avg = nn.AdaptiveAvgPool3d(pool_size)
        self.adaptive_max = nn.AdaptiveMaxPool3d(pool_size)

        # Compute total feature dimension:
        # For each stage: (avg + max) * channels * pool_size³
        spatial_elements = pool_size ** 3
        feature_dim = 0
        for s in stages:
            channels = variant.feature_size * self._STAGE_MULTIPLIERS[s]
            feature_dim += 2 * channels * spatial_elements  # avg + max concat

        self._feature_dim = feature_dim

        # Classification head
        hidden_dim = head_hidden_dim or max(feature_dim // 4, 128)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def feature_dim(self) -> int:
        """Total feature dimension fed to the classification head."""
        return self._feature_dim

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (stage 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (stage 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        """Whether the backbone is currently frozen."""
        return self._backbone_frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        # Extract multi-stage feature maps from SwinViT encoder
        features_list = self.backbone.swinViT(x)

        # Multi-scale pooling: adaptive avg + max pool per selected stage
        pooled_parts = []
        for s in self.stages:
            feat = features_list[s]  # (B, C_s, D_s, H_s, W_s)
            avg = self.adaptive_avg(feat).flatten(1)  # (B, C_s * pool³)
            mx = self.adaptive_max(feat).flatten(1)   # (B, C_s * pool³)
            pooled_parts.append(avg)
            pooled_parts.append(mx)

        pooled = torch.cat(pooled_parts, dim=1)  # (B, feature_dim)

        # Classification head
        return self.head(pooled)


# --- Factory for build_model compatibility ---


# --- Original VoCo-style Classifier (U-Net encoder-decoder + GAP) ---


class OriginalVoCoClassifier(nn.Module):
    """
    Classification model that mirrors the original VoCo paper architecture.

    Uses the full SwinViT encoder with UnetrBasicBlock encoders and
    UnetrUpBlock decoders, then applies global average pooling on the
    decoded output to produce a classification.

    This is essentially a U-Net-style encoder–decoder backbone where
    the final decoded feature map is reduced via adaptive avg pooling
    and fed to a linear classification head.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (1 for CT).
        variant: VoCoModelVariant enum (BASE, LARGE, HUGE).
        pretrained_path: Optional path to pretrained weights.
        freeze_backbone: If True, freeze backbone + encoder/decoder parameters.
        dropout: Dropout probability before the final linear layer.
        dropout_path_rate: Drop path rate for SwinViT (default 0.0).
        spatial_dims: Number of spatial dimensions (default 3).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        variant: VoCoModelVariant = VoCoModelVariant.HUGE,
        pretrained_path: Path | str | None = None,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        dropout_path_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.variant = variant
        self._backbone_frozen = False

        feature_size = variant.feature_size
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        norm_name = "instance"

        # --- SwinViT backbone ---
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            use_v2=True,
        )

        # --- Encoder blocks ---
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )

        # --- Decoder blocks ---
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_size, num_classes),
        )

        # --- Load pretrained weights into swinViT ---
        self._load_pretrained(variant, pretrained_path)

        if freeze_backbone:
            self.freeze_backbone()

    def _load_pretrained(self, variant, pretrained_path):
        """Load VoCo pretrained weights into the swinViT."""
        if pretrained_path is not None:
            weight_path = str(pretrained_path)
        else:
            weight_path = _download_voco_model(
                model_weight_name=variant.weight_filename,
                cache_dir=VOCO_PRETRAINED_WEIGHTS_DIR,
            )

        model_dict = torch.load(
            weight_path, map_location=torch.device("cpu"), weights_only=False
        )
        state_dict = model_dict.get("state_dict", model_dict)

        # Only load swinViT keys
        swin_prefix = "swinViT."
        swin_state = {}
        for k, v in state_dict.items():
            # Handle keys with or without the "swinViT." prefix
            if k.startswith(swin_prefix):
                swin_state[k[len(swin_prefix):]] = v
            elif k.startswith("module.swinViT."):
                swin_state[k[len("module.swinViT."):]] = v

        if not swin_state:
            # Fallback: try loading all keys that match swinViT's state_dict
            current = self.swinViT.state_dict()
            swin_state = {
                k: v for k, v in state_dict.items()
                if k in current and v.size() == current[k].size()
            }

        current = self.swinViT.state_dict()
        filtered = {
            k: v for k, v in swin_state.items()
            if k in current and v.size() == current[k].size()
        }
        current.update(filtered)
        self.swinViT.load_state_dict(current, strict=True)
        print(f"OriginalVoCoClassifier: loaded {len(filtered)}/{len(current)} swinViT params")

    def freeze_backbone(self) -> None:
        """Freeze swinViT + encoder/decoder parameters."""
        for module in [self.swinViT,
                       self.encoder1, self.encoder2, self.encoder3,
                       self.encoder4, self.encoder10,
                       self.decoder5, self.decoder4, self.decoder3,
                       self.decoder2, self.decoder1]:
            for param in module.parameters():
                param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, k: int = 32) -> torch.Tensor:
        """Pad spatial dims (D, H, W) so each is a multiple of k."""
        _, _, d, h, w = x.shape
        pad_d = (k - d % k) % k
        pad_h = (k - h % k) % k
        pad_w = (k - w % k) % k
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return x
        # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        b = x.size(0)

        # The original VoCo paper doubles the depth dimension.
        x_in = torch.cat([x, x], dim=2)

        # Pad to multiple of 32 so encoder downsampling / decoder upsampling
        # never produces mismatched spatial dims at skip connections.
        x_in = self._pad_to_multiple(x_in, k=32)

        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = self.head(out.view(b, -1))

        return out


# --- VoCoClassifierSeg: full U-Net + classification head + segmentation head ---


class VoCoClassifierSeg(nn.Module):
    """
    Dual-head model: classification + segmentation, built on the full VoCo U-Net.

    Identical architecture to ``OriginalVoCoClassifier`` (SwinViT encoder +
    UnetrBasicBlock encoders + UnetrUpBlock decoders). Two heads are branched
    from the full-resolution decoder output ``out``:

    - **Classification head**: global average pool → linear (same as OriginalVoCoClassifier)
    - **Segmentation head**: Conv3d(feature_size, num_seg_classes, 1) → trilinear
      interpolation back to the original input spatial size.

    ``forward()`` returns ``(cls_logits, seg_logits)`` where:
        - ``cls_logits``:  (B, num_classes)
        - ``seg_logits``:  (B, num_seg_classes, D, H, W)  — same D/H/W as input

    For FP crops the GT segmentation mask is all-zeros; the segmentation head
    is trained to predict *nothing* for false positives, which is the correct
    behaviour.

    Args:
        num_classes: Number of classification output classes.
        num_seg_classes: Number of segmentation output classes (default: 1 for binary).
        in_channels: Number of input channels (1 for CT).
        variant: VoCoModelVariant enum (BASE, LARGE, HUGE).
        pretrained_path: Optional path to pretrained weights.
        freeze_backbone: If True, freeze backbone + encoder/decoder at init.
        dropout: Dropout probability before the classification head.
        dropout_path_rate: Drop path rate for SwinViT (default 0.0).
        spatial_dims: Number of spatial dimensions (default 3).
    """

    def __init__(
        self,
        num_classes: int,
        num_seg_classes: int = 1,
        in_channels: int = 1,
        variant: VoCoModelVariant = VoCoModelVariant.BASE,
        pretrained_path: Path | str | None = None,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        dropout_path_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.variant = variant
        self._backbone_frozen = False

        feature_size = variant.feature_size
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        norm_name = "instance"

        # --- SwinViT backbone ---
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            use_v2=True,
        )

        # --- Encoder blocks ---
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3, stride=1,
            norm_name=norm_name, res_block=True,
        )

        # --- Decoder blocks ---
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=True,
        )

        # --- Classification head (same as OriginalVoCoClassifier) ---
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_size, num_classes),
        )

        # --- Segmentation head: 1×1 conv on full-resolution features ---
        self.seg_head = nn.Conv3d(feature_size, num_seg_classes, kernel_size=1)

        # --- Load pretrained weights into swinViT ---
        self._load_pretrained(variant, pretrained_path)

        if freeze_backbone:
            self.freeze_backbone()

    def _load_pretrained(self, variant, pretrained_path):
        """Load VoCo pretrained weights into the swinViT (identical to OriginalVoCoClassifier)."""
        if pretrained_path is not None:
            weight_path = str(pretrained_path)
        else:
            weight_path = _download_voco_model(
                model_weight_name=variant.weight_filename,
                cache_dir=VOCO_PRETRAINED_WEIGHTS_DIR,
            )

        model_dict = torch.load(
            weight_path, map_location=torch.device("cpu"), weights_only=False
        )
        state_dict = model_dict.get("state_dict", model_dict)

        swin_prefix = "swinViT."
        swin_state = {}
        for k, v in state_dict.items():
            if k.startswith(swin_prefix):
                swin_state[k[len(swin_prefix):]] = v
            elif k.startswith("module.swinViT."):
                swin_state[k[len("module.swinViT."):]] = v

        if not swin_state:
            current = self.swinViT.state_dict()
            swin_state = {
                k: v for k, v in state_dict.items()
                if k in current and v.size() == current[k].size()
            }

        current = self.swinViT.state_dict()
        filtered = {
            k: v for k, v in swin_state.items()
            if k in current and v.size() == current[k].size()
        }
        current.update(filtered)
        self.swinViT.load_state_dict(current, strict=True)
        print(f"VoCoClassifierSeg: loaded {len(filtered)}/{len(current)} swinViT params")

    def freeze_backbone(self) -> None:
        """Freeze swinViT + encoder/decoder parameters; keep heads trainable."""
        for module in [self.swinViT,
                       self.encoder1, self.encoder2, self.encoder3,
                       self.encoder4, self.encoder10,
                       self.decoder5, self.decoder4, self.decoder3,
                       self.decoder2, self.decoder1]:
            for param in module.parameters():
                param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, k: int = 32) -> torch.Tensor:
        """Pad spatial dims (D, H, W) so each is a multiple of k."""
        _, _, d, h, w = x.shape
        pad_d = (k - d % k) % k
        pad_h = (k - h % k) % k
        pad_w = (k - w % k) % k
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Tuple (cls_logits, seg_logits):
                cls_logits: (B, num_classes)
                seg_logits: (B, num_seg_classes, D, H, W)
        """
        b = x.size(0)
        D, H, W = x.shape[2], x.shape[3], x.shape[4]

        # Depth doubling trick (from the original VoCo paper)
        x_in = torch.cat([x, x], dim=2)
        x_in = self._pad_to_multiple(x_in, k=32)

        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        # out: (B, feature_size, D_padded, H_padded, W_padded)

        # --- Classification head ---
        cls_logits = self.head(F.adaptive_avg_pool3d(out, (1, 1, 1)).view(b, -1))

        # --- Segmentation head ---
        # 1×1 conv, then interpolate back to the original input spatial size
        seg_logits = self.seg_head(out)  # (B, num_seg_classes, D_padded, ...)
        seg_logits = F.interpolate(seg_logits, size=(D, H, W), mode="trilinear", align_corners=False)

        return cls_logits, seg_logits


def VoCoClassifierSeg_factory(cfg: dict, input_size, num_classes: int) -> VoCoClassifierSeg:
    """Factory function compatible with build_model(cfg, input_size, num_classes).

    Expected cfg keys (all optional except 'class'):
        class: "VoCoClassifierSeg"
        variant: "base" | "large" | "huge"  (default: "base")
        num_seg_classes: int  (default: 1)
        dropout: float  (default: 0.3)
        freeze_backbone: bool  (default: True)
        pretrained_path: str | None  (default: None, auto-downloads)
        dropout_path_rate: float  (default: 0.0)
    """
    variant = cfg.get("variant", "base")
    num_seg_classes = cfg.get("num_seg_classes", 1)
    dropout = cfg.get("dropout", 0.3)
    freeze_backbone = cfg.get("freeze_backbone", True)
    pretrained_path = cfg.get("pretrained_path", None)
    dropout_path_rate = cfg.get("dropout_path_rate", 0.0)

    return VoCoClassifierSeg(
        num_classes=num_classes,
        num_seg_classes=num_seg_classes,
        in_channels=input_size[0],
        variant=VoCoModelVariant(variant.lower()),
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        dropout_path_rate=dropout_path_rate,
    )


def OriginalVoCoClassifier_factory(cfg: dict, input_size, num_classes: int) -> OriginalVoCoClassifier:
    """Factory function compatible with build_model(cfg, input_size, num_classes).

    Expected cfg keys (all optional except 'class'):
        class: "OriginalVoCoClassifier"
        variant: "base" | "large" | "huge"  (default: "huge")
        dropout: float  (default: 0.3)
        freeze_backbone: bool  (default: True)
        pretrained_path: str | None  (default: None, auto-downloads)
        dropout_path_rate: float  (default: 0.0)
    """
    variant = cfg.get("variant", "huge")
    dropout = cfg.get("dropout", 0.3)
    freeze_backbone = cfg.get("freeze_backbone", True)
    pretrained_path = cfg.get("pretrained_path", None)
    dropout_path_rate = cfg.get("dropout_path_rate", 0.0)

    return OriginalVoCoClassifier(
        num_classes=num_classes,
        in_channels=input_size[0],
        variant=VoCoModelVariant(variant.lower()),
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        dropout_path_rate=dropout_path_rate,
    )


def VoCoClassifier_factory(cfg: dict, input_size, num_classes: int) -> VoCoClassifier:
    """Factory function compatible with build_model(cfg, input_size, num_classes).

    Reads model hyper-parameters from the config dict and returns a
    VoCoClassifier instance.

    Expected cfg keys (all optional except 'class'):
        class: "VoCoClassifier"  (used by build_model dispatch)
        variant: "base" | "large" | "huge"  (default: "huge")
        dropout: float  (default: 0.3)
        freeze_backbone: bool  (default: True)
        pretrained_path: str | None  (default: None, auto-downloads)
        head_hidden_dim: int | None  (default: None → feature_dim // 4)
        stages: list[int]  (default: [0, 1, 2])
        pool_size: int  (default: 2)
    """
    variant = cfg.get("variant", "huge")
    dropout = cfg.get("dropout", 0.3)
    freeze_backbone = cfg.get("freeze_backbone", True)
    pretrained_path = cfg.get("pretrained_path", None)
    head_hidden_dim = cfg.get("head_hidden_dim", None)
    stages = tuple(cfg.get("stages", [0, 1, 2]))
    pool_size = cfg.get("pool_size", 2)

    return VoCoClassifier(
        num_classes=num_classes,
        in_channels=input_size[0],  # (C, D, H, W)
        variant=VoCoModelVariant(variant.lower()),
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
        stages=stages,
        pool_size=pool_size,
    )


