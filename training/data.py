"""
PyTorch datasets over an export folder (tools/data.py export). Thin: the real
work is in augment.py and targets.py, which are tested without PyTorch.
"""

import os
from typing import Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from training.augment import crop_person, detector_sample, reid_sample
from training.targets import encode_center


class DetectorDataset(Dataset):
    def __init__(self, items: Sequence[dict], root: str, size: int = 320, train: bool = True, seed: int = 0):
        self.items, self.root, self.size, self.train = list(items), root, size, train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = cv2.imread(os.path.join(self.root, it["image"]))
        if img is None:
            raise FileNotFoundError(os.path.join(self.root, it["image"]))
        # a fresh random stream per (epoch, item): reproducible, different every epoch
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        rgb, boxes = detector_sample(img, it["boxes"], self.size, rng)
        heat, reg, logwh, ind, mask = encode_center(boxes, self.size)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()   # RGB 0..255
        return x, torch.from_numpy(heat), torch.from_numpy(reg), torch.from_numpy(logwh), \
            torch.from_numpy(ind), torch.from_numpy(mask)


class ReIDDataset(Dataset):
    """Crops are cut once and cached as small JPEGs next to the export, so each
    epoch reads 20 KB crops instead of decoding full frames."""

    def __init__(self, items: Sequence[dict], root: str, cache_dir: str, train: bool = True,
                 id_map: Optional[Dict[str, int]] = None, seed: int = 0):
        self.items, self.root, self.train, self.seed = list(items), root, train, seed
        self.epoch = 0
        self.id_map = id_map if id_map is not None else {k: n for n, k in enumerate(sorted({x["id"] for x in items}))}
        self.labels = [self.id_map.get(x["id"], -1) for x in self.items]
        os.makedirs(cache_dir, exist_ok=True)
        self.paths: List[str] = []
        by_image: Dict[str, List[int]] = {}
        for k, it in enumerate(self.items):
            by_image.setdefault(it["image"], []).append(k)
        self.paths = [""] * len(self.items)
        made = 0
        for image, idx in by_image.items():
            todo = []
            for k in idx:
                l, t, r, b = self.items[k]["box"]
                name = f"{image.replace('/', '_').replace('.jpg', '')}_{l}_{t}_{r}_{b}.jpg"
                self.paths[k] = os.path.join(cache_dir, name)
                if not os.path.exists(self.paths[k]):
                    todo.append(k)
            if todo:
                img = cv2.imread(os.path.join(root, image))
                if img is None:
                    continue
                for k in todo:
                    cv2.imwrite(self.paths[k], crop_person(img, self.items[k]["box"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    made += 1
        if made:
            print(f"  cached {made} new re-ID crops in {cache_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        crop = cv2.imread(self.paths[i])
        if crop is None:
            crop = np.zeros((256, 128, 3), np.uint8)
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        return torch.from_numpy(reid_sample(crop, rng)), self.labels[i]


class PKSampler(Sampler):
    """Batches of P identities x K crops each: the triplet loss needs several
    crops of the same person, and several people, in every batch."""

    def __init__(self, labels: Sequence[int], p: int = 16, k: int = 4, seed: int = 0):
        self.p, self.k, self.seed = p, k, seed
        self.by_id: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            if lab >= 0:
                self.by_id.setdefault(lab, []).append(i)
        self.ids = sorted(self.by_id)
        self.n_batches = max(1, sum(len(v) for v in self.by_id.values()) // (p * k))
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        p = min(self.p, len(self.ids))
        for _ in range(self.n_batches):
            batch: List[int] = []
            for ident in rng.choice(self.ids, p, replace=False):
                pool = self.by_id[int(ident)]
                batch += list(rng.choice(pool, self.k, replace=len(pool) < self.k))
            yield [int(b) for b in batch]
