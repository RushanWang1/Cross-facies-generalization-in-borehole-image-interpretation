import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.ops import DeformConv2d

class DeformConv2dPack(nn.Module):
    """
    DCNv1-style deformable conv: learns offsets and applies torchvision DeformConv2d.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=1):
        super().__init__()
        k = kernel_size
        self.offset = nn.Conv2d(
            in_ch, 2 * k * k,
            kernel_size=k, stride=stride, padding=padding
        )
        self.dcn = DeformConv2d(
            in_ch, out_ch,
            kernel_size=k, stride=stride, padding=padding, bias=bias
        )

        # Start as regular conv (zero offsets)
        nn.init.constant_(self.offset.weight, 0.0)
        nn.init.constant_(self.offset.bias, 0.0)

    def forward(self, x):
        off = self.offset(x)
        return self.dcn(x, off)

def replace_decoder_convs_with_dcn(unet_model: nn.Module, block_ids=None):
    """
    Replace 3x3 Conv2d inside SMP Unet decoder blocks with DeformConv2dPack.
    
    Args:
        unet_model: smp.Unet instance
        block_ids: iterable of decoder block indices to patch (e.g., [2,3,4]).
                  If None -> patch all blocks.
    """
    decoder = unet_model.decoder
    blocks = decoder.blocks  # ModuleList of DecoderBlock

    if block_ids is None:
        block_ids = range(len(blocks))

    for bi in block_ids:
        b = blocks[bi]

        # SMP DecoderBlock has conv1, conv2 of type Conv2dReLU
        for name in ["conv1", "conv2"]:
            if not hasattr(b, name):
                continue
            conv_relu = getattr(b, name)

            # Conv2dReLU typically has attribute .conv (nn.Conv2d)
            if not hasattr(conv_relu, "conv"):
                continue

            conv = conv_relu.conv
            if not isinstance(conv, nn.Conv2d):
                continue

            # Only replace the typical 3x3 unet convs
            k = conv.kernel_size
            p = conv.padding
            s = conv.stride
            d = conv.dilation
            g = conv.groups

            if k == (3, 3) and p == (1, 1) and s == (1, 1) and d == (1, 1):
                conv_relu.conv = DeformConv2dPack(
                    in_ch=conv.in_channels,
                    out_ch=conv.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(conv.bias is not None),
                    groups=g,
                )


class UNetWithPE(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=6, classes=1, pe_channels=3, use_dcn=False, dcn_decoder_blocks="deep", return_features =False  # "all", "deep", or list of indices
        ):
        """
        Args:
            in_channels: total number of input channels (RGB + PE)
            classes: number of output segmentation classes.
            pe_channels: how many PE channels to inject (e.g. depth=1 + angle=2).
        """
        super().__init__()

        # Use only RGB channels for encoder input
        self.rgb_channels = 3
        self.pe_channels = pe_channels
        self.return_features = return_features

        # Standard SMP U-Net (only RGB input)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.rgb_channels,  # only RGB
            classes=classes
        )

        # For each encoder skip stage, make a 1x1 conv for PE projection
        self.pe_projs = nn.ModuleList([
            nn.Conv2d(pe_channels, c, kernel_size=1) if c > 0 else None
            for c in self.unet.encoder.out_channels
        ]) 
        if use_dcn:
            n_blocks = len(self.unet.decoder.blocks)

            if dcn_decoder_blocks == "all":
                block_ids = list(range(n_blocks))
            elif dcn_decoder_blocks == "deep":
                # "deep" = closest to bottleneck (usually early indices in SMP decoder.blocks)
                # Patch first ~half blocks; adjust as you like.
                block_ids = list(range(max(1, n_blocks // 2)))
            elif isinstance(dcn_decoder_blocks, (list, tuple)):
                block_ids = list(dcn_decoder_blocks)
            else:
                raise ValueError("dcn_decoder_blocks must be 'all', 'deep', or a list of indices")

            replace_decoder_convs_with_dcn(self.unet, block_ids=block_ids)

    def forward(self, x):
        """
        Args:
            x: input tensor [B, in_channels, H, W] = [B, RGB+PE, H, W]
                RGB = first 3 channels, PE = remaining pe_channels
        """
        rgb = x[:, :3]
        pe_full = x[:, 3:]

        # Extract multi-scale features from encoder using RGB only
        features = self.unet.encoder(rgb)

        # Inject positional encoding into each encoder stage
        pe_feats = []
        for i, f in enumerate(features):
            if self.pe_projs[i] is None:
                pe_feats.append(f)   # skip PE injection for this stage
                continue
            _, _, h, w = f.shape
            pe_scaled = nn.functional.interpolate(
                pe_full, size=(h, w), mode="bilinear", align_corners=False
            )
            pe_proj = self.pe_projs[i](pe_scaled)
            f = f + pe_proj  # inject PE into encoder features
            pe_feats.append(f)

        # Decode with modified skip features
        decoder_output = self.unet.decoder(pe_feats)
        masks = self.unet.segmentation_head(decoder_output)
        if self.return_features:
            return masks, decoder_output                           # logits, features
        return masks

class UNetWithPE_BH(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=6, classes=1, pe_channels=3, use_dcn=False, dcn_decoder_blocks="deep", return_features =False,  # "all", "deep", or list of indices
                 n_boreholes=50, bh_embed_dim=8
        ):
        """
        Args:
            in_channels: total number of input channels (RGB + PE)
            classes: number of output segmentation classes.
            pe_channels: how many PE channels to inject (e.g. depth=1 + angle=2).
        """
        super().__init__()
        self.bh_embedding = nn.Embedding(n_boreholes, bh_embed_dim)

        # Use only RGB channels for encoder input
        self.rgb_channels = 3
        self.pe_channels = pe_channels
        self.return_features = return_features

        # Standard SMP U-Net (only RGB input)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.rgb_channels,  # only RGB
            classes=classes
        )

        # For each encoder skip stage, make a 1x1 conv for PE projection
        self.pe_projs = nn.ModuleList([
            nn.Conv2d(pe_channels + bh_embed_dim, c, kernel_size=1)
            for c in self.unet.encoder.out_channels
        ]) 
        if use_dcn:
            n_blocks = len(self.unet.decoder.blocks)

            if dcn_decoder_blocks == "all":
                block_ids = list(range(n_blocks))
            elif dcn_decoder_blocks == "deep":
                # "deep" = closest to bottleneck (usually early indices in SMP decoder.blocks)
                # Patch first ~half blocks; adjust as you like.
                block_ids = list(range(max(1, n_blocks // 2)))
            elif isinstance(dcn_decoder_blocks, (list, tuple)):
                block_ids = list(dcn_decoder_blocks)
            else:
                raise ValueError("dcn_decoder_blocks must be 'all', 'deep', or a list of indices")

            replace_decoder_convs_with_dcn(self.unet, block_ids=block_ids)

    def forward(self, x, bh_ids):
        """
        Args:
            x: input tensor [B, in_channels, H, W] = [B, RGB+PE, H, W]
                RGB = first 3 channels, PE = remaining pe_channels
        """
        rgb = x[:, :3]
        pe_full = x[:, 3:]

        # Get borehole embedding and broadcast to spatial dimensions
        bh_emb = self.bh_embedding(bh_ids)          # [B, bh_embed_dim]
        B, E   = bh_emb.shape
        _, _, H, W = x.shape
        bh_map = bh_emb.view(B, E, 1, 1).expand(B, E, H, W)  # [B, E, H, W]

        # Concatenate with PE
        pe_with_bh = torch.cat([pe_full, bh_map], dim=1)      # [B, 3+E, H, W]

        # Rest of forward unchanged — just use pe_with_bh instead of pe_full
        features = self.unet.encoder(rgb)
        pe_feats = []
        for i, f in enumerate(features):
            _, _, h, w = f.shape
            pe_scaled = F.interpolate(pe_with_bh, (h, w), mode="bilinear", align_corners=False)
            pe_proj   = self.pe_projs[i](pe_scaled)
            f         = f + pe_proj
            pe_feats.append(f)

        # # Extract multi-scale features from encoder using RGB only
        # features = self.unet.encoder(rgb)

        # # Inject positional encoding into each encoder stage
        # pe_feats = []
        # for i, f in enumerate(features):
        #     _, _, h, w = f.shape
        #     pe_scaled = nn.functional.interpolate(
        #         pe_full, size=(h, w), mode="bilinear", align_corners=False
        #     )
        #     pe_proj = self.pe_projs[i](pe_scaled)
        #     f = f + pe_proj  # inject PE into encoder features
        #     pe_feats.append(f)

        # Decode with modified skip features
        decoder_output = self.unet.decoder(pe_feats)
        masks = self.unet.segmentation_head(decoder_output)
        if self.return_features:
            return masks, decoder_output                           # logits, features
        return masks


class PriorAttentionGate(nn.Module):
    """
    Takes the structural mask, resizes it to match the feature map,
    and learns a weighting map to gate the features.
    """
    def __init__(self):
        super().__init__()
        # A simple convolution to allow the network to learn 
        # best how to use the mask (e.g., scaling, inverting, or sharpening it)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, features, mask):
        # 1. Resize the structural mask to match current feature map resolution
        mask_resized = F.interpolate(mask, size=features.shape[2:], mode="bilinear", align_corners=False)
        
        # 2. Generate Attention Map (0 to 1)
        attn_weights = self.mask_conv(mask_resized)
        
        # 3. Apply Attention
        # We add +0.1 to ensure features aren't completely killed if the mask is slightly off.
        # This makes it "Soft Attention".
        return features * (attn_weights + 0.1)

class UNetWithPEAndAttention(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=6, classes=1, pe_channels=3):
        """
        Args:
            in_channels: total inputs in tensor (RGB + PE). 
                         Note: Structural mask is passed separately.
        """
        super().__init__()

        self.rgb_channels = 3
        self.pe_channels = pe_channels

        # 1. Standard SMP U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.rgb_channels, 
            classes=classes
        )

        # 2. PE Projections 
        # We use the encoder's out_channels to define the projection size
        encoder_channels = self.unet.encoder.out_channels
        self.pe_projs = nn.ModuleList([
            nn.Conv2d(pe_channels, c, kernel_size=1)
            for c in encoder_channels
        ])

        # 3. NEW: Attention Gates
        # We create one gate for each stage of the encoder
        self.att_gates = nn.ModuleList([
            PriorAttentionGate()
            for _ in encoder_channels
        ])

    def forward(self, x, structural_mask):
        """
        Args:
            x: [B, RGB+PE, H, W]
            structural_mask: [B, 1, H, W] -> The Gaussian blurred sine wave heatmap
        """
        rgb = x[:, :3]
        pe_full = x[:, 3:]

        # A. Extract features (Texture)
        features = self.unet.encoder(rgb)

        # B. Inject Information into Skip Connections
        modified_features = []
        
        # smp encoder output is a list of tensors, usually 6 levels for ResNet
        for i, f in enumerate(features):
            h, w = f.shape[2], f.shape[3]
            
            # --- 1. Position Embedding Injection (Your Code) ---
            pe_scaled = F.interpolate(
                pe_full, size=(h, w), mode="bilinear", align_corners=False
            )
            pe_proj = self.pe_projs[i](pe_scaled)
            
            # Fuse Texture + Position
            f_fused = f + pe_proj 

            # --- 2. Structural Attention (New Code) ---
            # We gate the Fused features using the Structural Mask
            f_attended = self.att_gates[i](f_fused, structural_mask)
            
            modified_features.append(f_attended)

        # C. Decode
        decoder_output = self.unet.decoder(modified_features)
        masks = self.unet.segmentation_head(decoder_output)

        return masks

# --- Test the model ---
if __name__ == "__main__":
    model = UNetWithPE()
    x = torch.randn(1, 6, 256, 256)  # [RGB(3) + PE(3)]
    y = model(x)
    print("Output shape:", y.shape)


# class UNetWithPE(nn.Module):
#     def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
#                  in_channels=6, classes=1, pe_channels=3):
#         """
#         Args:
#             in_channels: number of input channels (RGB+coords=6 in your case).
#             classes: number of output segmentation classes.
#             pe_channels: how many PE channels to inject (depth=1 + angle=2).
#         """
#         super().__init__()
#         self.unet = smp.Unet(
#             encoder_name=encoder_name,
#             encoder_weights=encoder_weights,
#             in_channels=in_channels,
#             classes=classes
#         )
#         self.pe_channels = pe_channels

#         # For each skip stage in the encoder, make a 1x1 conv to align channels
#         # self.unet.encoder has .out_channels list like [64, 128, 256, 512]
#         self.pe_projs = nn.ModuleList([
#             nn.Conv2d(pe_channels, c, kernel_size=1)
#             for c in self.unet.encoder.out_channels
#         ])

#     def forward(self, x):
#         """
#         Args:
#             x: input image tensor, shape [B, in_channels, H, W]
#             pe_full: positional encodings for the full input (B, pe_channels, H, W).
#                      (If None, skips injection.)
#         """
#         # Get encoder feature maps
#         features = self.unet.encoder(x)  # list of feature maps at multiple scales
#         pe_full = x[:,3:]

#         if pe_full is not None:
#             B, _, H, W = x.shape
#             # Resize PE for each scale and inject into skip features
#             pe_feats = []
#             for i, f in enumerate(features):
#                 _, _, h, w = f.shape
#                 pe_scaled = nn.functional.interpolate(pe_full, size=(h, w),
#                                                       mode="bilinear", align_corners=False)
#                 pe_proj = self.pe_projs[i](pe_scaled)
#                 f = f + pe_proj  # inject PE
#                 pe_feats.append(f)
#             features = pe_feats

#         # Decode with skip connections
#         decoder_output = self.unet.decoder(features)
#         masks = self.unet.segmentation_head(decoder_output)
#         return masks



# if __name__ == "__main__":
#     model = UNetWithPE().cuda()
#     inp = torch.randn(1, 6, 512, 512).cuda()
#     out = model(inp)
#     print(out.shape)  # should be [1, 1, 512, 512]

#     # Visualize the model graph
#     dot = make_dot(out, params=dict(model.named_parameters()))
#     dot.format = 'png'
#     dot.render('unet_with_pe_graph')