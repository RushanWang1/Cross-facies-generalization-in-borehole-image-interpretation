# This file will be used to generalize the dataset for borehole segmentation tasks.

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple
import cv2
import random
import numpy as np
from glob import glob   
import os
import random
import torch.nn.functional as F
from data_utils import extract_traces_graph
from config import ModelCfg


class TectonicPatchDataset_multiclass(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        soft_paths=None,
        patch_sizes=(256,),
        samples_per_image=200,        # samples for a borehole of reference length
        reference_length_px=20000,     # px height that receives `samples_per_image`;
                                      # None → median height of this dataset
        min_samples=20,               # floor for very short boreholes
        max_samples=400,              # cap for very long boreholes
        trace_prob=0.7,
        transform=None,
        add_coords=True,
        add_soft_channel=False,
        preload=True,
        background_idx=0,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.soft_paths = soft_paths
        self.patch_sizes = list(patch_sizes)
        self.trace_prob = float(trace_prob)
        self.transform = transform
        self.add_coords = add_coords
        self.add_soft_channel = add_soft_channel
        self.preload = preload
        self.background_idx = background_idx

        if preload:
            self.images = [
                cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                for p in image_paths
            ]
            self.masks = [
                cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                for p in mask_paths
            ]
            if soft_paths is not None:
                self.softs = [
                    cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    for p in soft_paths
                ]
        else:
            self.images, self.masks, self.softs = None, None, None

        for i, (ip, mp) in enumerate(zip(image_paths, mask_paths)):
            img  = self.images[i]
            mask = self.masks[i]
            if img.shape[:2] != mask.shape[:2]:
                print(f"SIZE MISMATCH [{i}]: image={img.shape[:2]} mask={mask.shape[:2]}")
                print(f"  image: {ip}")
                print(f"  mask:  {mp}")

        # ---- Height-proportional sample counts ----
        # A borehole of reference_length_px gets `samples_per_image` samples;
        # others scale linearly with their image height, clipped to
        # [min_samples, max_samples].
        heights = [
            (self.masks[i] if preload
             else cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)).shape[0]
            for i in range(len(image_paths))
        ]
        if reference_length_px is None:
            reference_length_px = float(np.median(heights))
        self.reference_length_px = reference_length_px

        self.samples_per_image_list = [
            int(np.clip(round(samples_per_image * h / reference_length_px),
                        min_samples, max_samples))
            for h in heights
        ]
        # cum_samples[i] = first global index belonging to image i
        self.cum_samples = np.concatenate(
            [[0], np.cumsum(self.samples_per_image_list)]
        ).astype(np.int64)

        print(f"[Dataset] Height-proportional sampling: "
              f"ref={reference_length_px:.0f}px, "
              f"floor={min_samples}, cap={max_samples}")
        print(f"[Dataset] {len(image_paths)} images, "
              f"{self.cum_samples[-1]} samples/epoch "
              f"(per-image range {min(self.samples_per_image_list)}"
              f"-{max(self.samples_per_image_list)})")

        # ---- Precompute trace coordinates (BIG SPEEDUP) ----
        self.trace_coords = []
        self.current_epoch = 0
        for i in range(len(image_paths)):
            m = self.masks[i] if preload else cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            ys, xs = np.nonzero(m != self.background_idx)
            self.trace_coords.append((ys.astype(np.int32), xs.astype(np.int32)))

    def __len__(self):
        return int(self.cum_samples[-1])

    def _img_idx_for(self, idx):
        """Map a global sample index to its image index via cumulative counts."""
        return int(np.searchsorted(self.cum_samples, idx, side="right") - 1)

    def _load(self, img_idx):
        if self.preload:
            image = self.images[img_idx]
            mask = self.masks[img_idx]
        else:
            image = cv2.cvtColor(
                cv2.imread(self.image_paths[img_idx], cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
        return image, mask

    def __getitem__(self, idx):
        img_idx = self._img_idx_for(idx)          # ← was: idx // self.samples_per_image
        image, mask = self._load(img_idx)

        h, w = mask.shape
        valid_patch_sizes = [s for s in self.patch_sizes if (s <= w and s <= h)]
        ps = valid_patch_sizes[np.random.randint(len(valid_patch_sizes))] if valid_patch_sizes else min(self.patch_sizes)

        ys, xs = self.trace_coords[img_idx]
        use_trace = (np.random.rand() < self.trace_prob) and (xs.size > 0)

        if use_trace:
            j = np.random.randint(xs.size)
            center_x, center_y = int(xs[j]), int(ys[j])
        else:
            cx_min, cx_max = ps // 2, w - ps // 2
            cy_min, cy_max = ps // 2, h - ps // 2
            if cx_max <= cx_min or cy_max <= cy_min:
                image_patch = cv2.resize(image, (ps, ps), interpolation=cv2.INTER_LINEAR)
                mask_patch = cv2.resize(mask, (ps, ps), interpolation=cv2.INTER_NEAREST)
                x1, y1 = 0, 0
                return self._finalize(image_patch, mask_patch, x1, y1, h, w, ps)

            center_x = np.random.randint(cx_min, cx_max)
            center_y = np.random.randint(cy_min, cy_max)

        x1 = max(0, center_x - ps // 2)
        y1 = max(0, center_y - ps // 2)
        x2, y2 = x1 + ps, y1 + ps

        if x2 > w:
            x2 = w
            x1 = w - ps
        if y2 > h:
            y2 = h
            y1 = h - ps

        image_patch = image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        if image_patch.shape[0] != ps or image_patch.shape[1] != ps:
            image_patch = cv2.resize(image_patch, (ps, ps), interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask_patch, (ps, ps), interpolation=cv2.INTER_NEAREST)
            x1, y1 = 0, 0

        return self._finalize(image_patch, mask_patch, x1, y1, h, w, ps)

    def _finalize(self, image_patch, mask_patch, x1, y1, h, w, ps):
        # ... unchanged from your current version ...
        if self.transform is not None:
            augmented = self.transform(image=image_patch, mask=mask_patch)
            image_t = augmented["image"]
            mask_t = augmented["mask"]
        else:
            image_t = torch.from_numpy(image_patch).permute(2, 0, 1).contiguous().float() / 255.0
            mask_t = torch.from_numpy(mask_patch).contiguous().long()

        if self.add_coords:
            ph, pw = image_t.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.arange(ph, dtype=torch.float32),
                torch.arange(pw, dtype=torch.float32),
                indexing="ij"
            )
            yy = (yy + y1) / h
            xx = (xx + x1) / w
            depth_enc = yy.unsqueeze(0)
            angle_enc = torch.stack(
                [torch.sin(2 * torch.pi * xx), torch.cos(2 * torch.pi * xx)],
                dim=0
            )
            pe = torch.cat([depth_enc, angle_enc], dim=0)
            image_t = torch.cat([image_t, pe], dim=0)

        return image_t, mask_t

class TectonicPatchDataset_multiclass_fixedsample(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        soft_paths=None,
        patch_sizes=(256,),
        samples_per_image=10,
        trace_prob=0.7,
        transform=None,
        add_coords=True,
        add_soft_channel=False,
        preload=True,
        background_idx=0,
        # borehole_ids = None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.soft_paths = soft_paths
        self.patch_sizes = list(patch_sizes)
        self.samples_per_image = samples_per_image
        self.trace_prob = float(trace_prob)
        self.transform = transform
        self.add_coords = add_coords
        self.add_soft_channel = add_soft_channel
        self.preload = preload
        self.background_idx = background_idx
        

        if preload:
            self.images = [
                cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                for p in image_paths
            ]
            self.masks = [
                cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                for p in mask_paths
            ]
            if soft_paths is not None:
                self.softs = [
                    cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    for p in soft_paths
                ]
        else:
            self.images, self.masks, self.softs = None, None, None

        for i, (ip, mp) in enumerate(zip(image_paths, mask_paths)):
            img  = self.images[i]
            mask = self.masks[i]
            if img.shape[:2] != mask.shape[:2]:
                print(f"SIZE MISMATCH [{i}]: image={img.shape[:2]} mask={mask.shape[:2]}")
                print(f"  image: {ip}")
                print(f"  mask:  {mp}")
        # ---- Precompute trace coordinates (BIG SPEEDUP) ----
        # store as (ys, xs) arrays per image
        self.trace_coords = []
        self.current_epoch = 0
        for i in range(len(image_paths)):
            m = self.masks[i] if preload else cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            ys, xs = np.nonzero(m != self.background_idx)
            self.trace_coords.append((ys.astype(np.int32), xs.astype(np.int32)))
        # Map each image path to an integer borehole ID
        # if borehole_ids is not None:
        #     self.borehole_ids = borehole_ids   # list of ints, one per image
        # else:
        #     # Auto-assign — each image gets its own ID
        #     self.borehole_ids = list(range(len(image_paths)))


    def __len__(self):
        return len(self.image_paths) * self.samples_per_image

    def _load(self, img_idx):
        if self.preload:
            image = self.images[img_idx]
            mask = self.masks[img_idx]
            # soft = self.softs[img_idx]
        else:
            image = cv2.cvtColor(cv2.imread(self.image_paths[img_idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
            # soft = cv2.imread(self.soft_paths[img_idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return image, mask, # soft

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        # image, mask, soft = self._load(img_idx)
        image, mask = self._load(img_idx)

        h, w = mask.shape
        valid_patch_sizes = [s for s in self.patch_sizes if (s <= w and s <= h)]
        ps = valid_patch_sizes[np.random.randint(len(valid_patch_sizes))] if valid_patch_sizes else min(self.patch_sizes)

        # ORIGINAL sample method
        ys, xs = self.trace_coords[img_idx]
        # Original fixed use_trace:
        use_trace = (np.random.rand() < self.trace_prob) and (xs.size > 0)

        # New decay-based use_trace:
        # decay     = max(0.3, self.trace_prob - self.current_epoch * 0.002)
        # use_trace = (np.random.rand() < decay) and (xs.size > 0)
        
        if use_trace:
            j = np.random.randint(xs.size)
            center_x, center_y = int(xs[j]), int(ys[j])
        else:
            cx_min, cx_max = ps // 2, w - ps // 2
            cy_min, cy_max = ps // 2, h - ps // 2
            if cx_max <= cx_min or cy_max <= cy_min:
                image_patch = cv2.resize(image, (ps, ps), interpolation=cv2.INTER_LINEAR)
                mask_patch = cv2.resize(mask, (ps, ps), interpolation=cv2.INTER_NEAREST)
                # soft_patch = cv2.resize(soft, (ps, ps), interpolation=cv2.INTER_LINEAR)
                x1, y1 = 0, 0
                return self._finalize(image_patch, mask_patch, x1, y1, h, w, ps)

            center_x = np.random.randint(cx_min, cx_max)
            center_y = np.random.randint(cy_min, cy_max)

        x1 = max(0, center_x - ps // 2)
        y1 = max(0, center_y - ps // 2)
        x2, y2 = x1 + ps, y1 + ps

        if x2 > w:
            x2 = w
            x1 = w - ps
        if y2 > h:
            y2 = h
            y1 = h - ps

        image_patch = image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]
        # soft_patch = soft[y1:y2, x1:x2]

        if image_patch.shape[0] != ps or image_patch.shape[1] != ps:
            image_patch = cv2.resize(image_patch, (ps, ps), interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask_patch, (ps, ps), interpolation=cv2.INTER_NEAREST)
            # soft_patch = cv2.resize(soft_patch, (ps, ps), interpolation=cv2.INTER_LINEAR)
            x1, y1 = 0, 0

        return self._finalize(image_patch, mask_patch,x1, y1, h, w, ps)

    def _finalize(self, image_patch, mask_patch, x1, y1, h, w, ps):
        if self.transform is not None:
            augmented = self.transform(image=image_patch, mask=mask_patch)
            image_t = augmented["image"]
            mask_t = augmented["mask"]
            # soft_t = augmented["soft"]
            # if isinstance(soft_t, np.ndarray):
            #     soft_t = torch.from_numpy(soft_t).float()
        else:
            image_t = torch.from_numpy(image_patch).permute(2, 0, 1).contiguous().float() / 255.0
            mask_t = torch.from_numpy(mask_patch).contiguous().long()
            # soft_t = torch.from_numpy(soft_patch).contiguous().float()

        # if soft_t.ndim == 2:
        #     soft_t = soft_t.unsqueeze(0)

        if self.add_coords:
            ph, pw = image_t.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.arange(ph, dtype=torch.float32),
                torch.arange(pw, dtype=torch.float32),
                indexing="ij"
            )
            # MAX_DEPTH_PX = 100000
            # yy = (yy + y1) / MAX_DEPTH_PX
            yy = (yy + y1) / h
            xx = (xx + x1) / w
            depth_enc = yy.unsqueeze(0)
            angle_enc = torch.stack(
                [torch.sin(2 * torch.pi * xx), torch.cos(2 * torch.pi * xx)],
                dim=0
            )
            pe = torch.cat([depth_enc, angle_enc], dim=0)
            image_t = torch.cat([image_t, pe], dim=0)

        return image_t, mask_t
        
    

class HardNegNPZDatasetWithPE(Dataset):
    def __init__(self, npz_dir, transform=None):
        self.files = sorted(
            os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        image = d["image"]   # uint8 [H,W,3]
        mask  = d["mask"]    # uint8 [H,W]

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            rgb = out["image"]           # torch [3,H,W], already normalized
            mask = out["mask"].long()
        else:
            rgb = torch.from_numpy(image).permute(2,0,1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        H, W = rgb.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij"
        )
        yy = yy / float(H)
        xx = xx / float(W)
        depth = yy.unsqueeze(0)
        angle = torch.stack([torch.sin(2*torch.pi*xx), torch.cos(2*torch.pi*xx)], dim=0)
        pe = torch.cat([depth, angle], dim=0)          # [3,H,W]

        x6 = torch.cat([rgb, pe], dim=0)               # [6,H,W]
        return x6, mask

class TectonicDualPathDataset(Dataset):
    def __init__(self, image_paths, mask_paths,
                 patch_sizes=[256],
                 samples_per_image=10,
                 patches_per_full=4,
                 global_downsample=16,
                 trace_prob=0.7,
                 transform=None,
                 add_coords=True,
                 preload=True):
        """
        For BoreholeDualPathSwin model.

        Args:
            image_paths: list of borehole image paths.
            mask_paths: corresponding binary mask paths.
            patch_sizes: list of local patch sizes.
            samples_per_image: number of patch batches per image per epoch.
            patches_per_full: number of local patches per full image.
            global_downsample: how much to downsample the full image for Swin.
            trace_prob: probability to sample around fractures.
            transform: Albumentations transforms (with ToTensorV2).
            add_coords: whether to append 3 PE channels to each patch.
            preload: whether to keep all images in memory.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_sizes = patch_sizes
        self.samples_per_image = samples_per_image
        self.patches_per_full = patches_per_full
        self.global_downsample = global_downsample
        self.trace_prob = trace_prob
        self.transform = transform
        self.add_coords = add_coords
        self.preload = preload

        if preload:
            self.images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]
            self.masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
        else:
            self.images, self.masks = None, None

    def __len__(self):
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx):       

        img_idx = idx // self.samples_per_image

        # --- Load image/mask ---
        if self.preload:
            image = self.images[img_idx].copy()
            mask = self.masks[img_idx].copy()
        else:
            image = cv2.cvtColor(cv2.imread(self.image_paths[img_idx]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)

        H, W = mask.shape
        valid_patch_sizes = [s for s in self.patch_sizes if s <= max(H, W)]
        ps = random.choice(valid_patch_sizes) if valid_patch_sizes else min(self.patch_sizes)

        patches, masks = [], []

        for _ in range(self.patches_per_full):
            # --- Pick patch ---
            if not valid_patch_sizes:
                ps = min(self.patch_sizes)
                img_patch = cv2.resize(image, (ps, ps))
                mask_patch = cv2.resize(mask, (ps, ps), interpolation=cv2.INTER_NEAREST)
                x1, y1 = 0, 0
            else:
                ps = random.choice(valid_patch_sizes)
                for _ in range(10):  # try to find valid patch
                    if random.random() < self.trace_prob:
                        ys, xs = np.where(mask > 0)
                        if len(xs) == 0:
                            break
                        idx2 = random.randint(0, len(xs) - 1)
                        center_x, center_y = xs[idx2], ys[idx2]
                    else:
                        center_x = random.randint(ps // 2, W - ps // 2)
                        center_y = random.randint(ps // 2, H - ps // 2)

                    x1 = max(0, center_x - ps // 2)
                    y1 = max(0, center_y - ps // 2)
                    x2 = x1 + ps
                    y2 = y1 + ps

                    if x2 <= W and y2 <= H:
                        img_patch = image[y1:y2, x1:x2]
                        mask_patch = mask[y1:y2, x1:x2]
                        break
                else:
                    # fallback: resize whole image
                    img_patch = cv2.resize(image, (ps, ps))
                    mask_patch = cv2.resize(mask, (ps, ps), interpolation=cv2.INTER_NEAREST)
                    x1, y1 = 0, 0
            
            # --- Albumentations transform (if given) ---
            if self.transform:
                augmented = self.transform(image=img_patch, mask=mask_patch)
                img_patch = augmented["image"]
                mask_patch = augmented["mask"]

            # --- Convert to torch if numpy ---
            if isinstance(img_patch, np.ndarray):
                img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.
            if isinstance(mask_patch, np.ndarray):
                mask_patch = torch.from_numpy(mask_patch).long()

            # --- Pad to exact patch size (just in case) ---
            _, ph, pw = img_patch.shape
            if ph < ps or pw < ps:
                pad_h = ps - ph
                pad_w = ps - pw
                img_patch = F.pad(img_patch, (0, pad_w, 0, pad_h), mode="reflect")
                mask_patch = F.pad(mask_patch.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)

            # --- Add positional encoding (depth + sinθ + cosθ) ---
            if self.add_coords:
                yy, xx = torch.meshgrid(
                    torch.arange(ps, dtype=torch.float32),
                    torch.arange(ps, dtype=torch.float32),
                    indexing="ij",
                )
                yy = (yy + y1) / H  # normalized depth
                xx = (xx + x1) / W  # normalized azimuth
                depth_enc = yy.unsqueeze(0)
                angle_enc = torch.stack(
                    [torch.sin(2 * torch.pi * xx), torch.cos(2 * torch.pi * xx)], dim=0
                )
                pe = torch.cat([depth_enc, angle_enc], dim=0)
                img_patch = torch.cat([img_patch, pe], dim=0)

            patches.append(img_patch)
            masks.append(mask_patch)

        # --- Stack patches for this borehole ---
        patches = torch.stack(patches, dim=0)  # (Np, C, H, W)
        masks = torch.stack(masks, dim=0)      # (Np, H, W)

        # --- Build downsampled global image ---
        global_img = cv2.resize(
            image,
            (W // self.global_downsample, H // self.global_downsample),
            interpolation=cv2.INTER_AREA,
        )
        global_img = torch.from_numpy(global_img).permute(2, 0, 1).float() / 255.
        
        trace_target = extract_traces_graph(mask, num_points=ModelCfg.detr_P)

        return {
            "full_img": global_img,   # (3, H', W')
            "patches": patches,       # (Np, C, ps, ps)
            "masks": masks,           # (Np, ps, ps)
            "trace_target": trace_target  # {'labels': Tensor, 'points': Tensor
        }


    def _sample_center(self, mask, ps, W, H):
        """Sample a center with a bias towards fracture areas."""
        if random.random() < self.trace_prob:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                i = random.randint(0, len(xs) - 1)
                return xs[i], ys[i]
        return random.randint(ps // 2, W - ps // 2), random.randint(ps // 2, H - ps // 2)
    
def extract_patches_from_full(full_img_np, patch_size, patches_per_full):
    H, W = full_img_np.shape[:2]
    coords = []
    patches = []

    # random uniform sampling across width
    xs = np.linspace(0, W-patch_size, patches_per_full).astype(int)
    ys = np.linspace(0, H-patch_size, patches_per_full).astype(int)

    for x in xs:
        for y in ys:
            patch = full_img_np[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((x, y))

    return np.array(patches), coords
