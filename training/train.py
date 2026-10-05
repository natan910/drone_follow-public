"""
One training entry point for all our models. Same code on the Mac (Apple GPU,
"mps"), on an NVIDIA box (AWS g4dn, "cuda") or plain CPU.

    # prove the whole pipeline works on this machine (synthetic data, ~2 min)
    python -m training.train smoke

    # our person detector, from our own footage (export: tools/data.py export --out exports/v1)
    python -m training.train detector --data exports/v1 --name det_v1

    # our re-ID net; starting its backbone from the detector's helps a lot
    python -m training.train reid --data exports/v1 --name reid_v1 --init runs/det_v1/best.pt

Each run writes runs/<name>/:
    last.pt, best.pt         checkpoints (best = lowest eval loss; last if no eval data)
    person_own.onnx          detector, ready for --person-detector own --own-model ...
    reid_own.onnx            re-ID net, ready for --reid onnx|fused --reid-model ...
    model_card.json          what it was trained on, how, when (provenance)
    log.jsonl                losses per epoch
Then benchmark it (tools/benchmark.py) before copying it into models/.
"""

import argparse
import json
import os
import platform as _platform
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset.export import load, root_of  # noqa: E402
from training.data import DetectorDataset, PKSampler, ReIDDataset  # noqa: E402
from training.models import (CenterNet, CenterNetExport, ModelEMA, ReIDExport, ReIDNet,  # noqa: E402
                             detector_loss, load_backbone, reid_loss)


# ---- plumbing ------------------------------------------------------------------------

def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * t)))


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except Exception:
        return None


def write_card(run_dir: str, **fields) -> None:
    card = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
            "torch": torch.__version__, "machine": _platform.platform(), **fields}
    with open(os.path.join(run_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=1)


def log_line(run_dir: str, row: dict) -> None:
    with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def loader(ds, batch: int, shuffle: bool, workers: int, device: torch.device, **kw) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=shuffle, **kw)


def save_ckpt(path: str, model: torch.nn.Module, epoch: int, extra: Dict) -> None:
    # tensors and plain numbers only: loads with torch.load(weights_only=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, **extra}, path)


# ---- ONNX export + check ---------------------------------------------------------------

def export_onnx(module: torch.nn.Module, dummy: torch.Tensor, path: str, dynamic_batch: bool) -> None:
    module = module.cpu().eval()
    kw = dict(input_names=["images"], output_names=["out"], opset_version=12, do_constant_folding=True)
    if dynamic_batch:
        kw["dynamic_axes"] = {"images": {0: "n"}, "out": {0: "n"}}
    try:
        torch.onnx.export(module, dummy, path, dynamo=False, **kw)    # classic exporter: opset 12, cv2-friendly
    except TypeError:                                                   # torch without the dynamo flag
        torch.onnx.export(module, dummy, path, **kw)


def check_with_cv2(module: torch.nn.Module, dummy: torch.Tensor, path: str) -> float:
    """Load the ONNX file the way the drone does (cv2.dnn) and compare with
    PyTorch on the same input. Returns the max absolute difference."""
    import cv2
    net = cv2.dnn.readNetFromONNX(path)
    net.setInput(dummy.numpy())
    got = net.forward()
    with torch.no_grad():
        want = module.cpu().eval()(dummy).numpy()
    return float(np.abs(got.reshape(want.shape) - want).max())


# ---- detector --------------------------------------------------------------------------

def train_detector(a) -> str:
    device = pick_device(a.device)
    root = root_of(a.data, a.root)
    tr_items, ev_items = load(a.data, "detector_train.json"), load(a.data, "detector_eval.json")
    if not tr_items:
        raise SystemExit(f"no training frames in {a.data}: run tools/data.py export")
    if a.size % 32:
        raise SystemExit(f"--size must be a multiple of 32 (got {a.size})")
    run_dir = os.path.join(a.runs, a.name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"detector: {len(tr_items)} train / {len(ev_items)} eval frames, device {device}, size {a.size}")

    model = CenterNet(width=a.width).to(device)
    if a.init:
        print(f"  backbone from {a.init}: {len(load_backbone(model, a.init))} tensors")
    ema = ModelEMA(model)
    tr = DetectorDataset(tr_items, root, a.size, train=True, seed=a.seed)
    ev = DetectorDataset(ev_items, root, a.size, train=False)
    tl = loader(tr, a.batch, True, a.workers, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    total, step, best = a.epochs * max(1, len(tl)), 0, float("inf")

    for epoch in range(a.epochs):
        model.train()
        tr.epoch = epoch
        t0, sums, n = time.time(), {}, 0
        for x, heat, reg, logwh, ind, mask in tl:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, a.lr, warmup=min(500, total // 10 + 1))
            x, heat, reg, logwh, ind, mask = (t.to(device, non_blocking=True) for t in (x, heat, reg, logwh, ind, mask))
            loss, parts = detector_loss(model(x), heat, reg, logwh, ind, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ema.update(model)
            step += 1
            n += 1
            for k, v in {"loss": float(loss), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v
            if a.max_steps and step >= a.max_steps:
                break
        row = {"epoch": epoch, **{k: round(v / max(1, n), 4) for k, v in sums.items()},
               "s": round(time.time() - t0, 1)}
        if ev_items:
            row["eval_loss"] = round(eval_detector(ema.ema, ev, a.batch, a.workers, device), 4)
        print(f"  epoch {epoch + 1}/{a.epochs} {row}")
        log_line(run_dir, row)
        extra = {"width": a.width, "size": a.size}
        save_ckpt(os.path.join(run_dir, "last.pt"), ema.ema, epoch, extra)
        score = row.get("eval_loss", row["loss"])
        if score < best:
            best = score
            save_ckpt(os.path.join(run_dir, "best.pt"), ema.ema, epoch, extra)
        if a.max_steps and step >= a.max_steps:
            break

    best_model = CenterNet(width=a.width)
    best_model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")["model"])
    onnx_path = os.path.join(run_dir, "person_own.onnx")
    wrapper = CenterNetExport(best_model)
    dummy = torch.rand(1, 3, a.size, a.size) * 255
    export_onnx(wrapper, dummy, onnx_path, dynamic_batch=False)
    diff = check_with_cv2(wrapper, dummy, onnx_path)
    print(f"  exported {onnx_path}; cv2.dnn vs torch max diff {diff:.2e} {'OK' if diff < 1e-3 else 'CHECK THIS'}")
    manifest = load(a.data, "manifest.json")
    write_card(run_dir, model="person_own", kind="detector (CenterNet-style, ours)",
               input=f"1x3x{a.size}x{a.size} RGB 0..255", output="1x5xH/4xW/4",
               width=a.width, epochs=a.epochs, lr=a.lr, batch=a.batch, device=str(device),
               init=a.init or "scratch (no pretrained weights)",
               data=a.data, dataset_manifest=manifest, best_score=best, cv2_max_diff=diff,
               usage=f"--person-detector own --own-model {onnx_path}  (own_input_size = {a.size})")
    return onnx_path


@torch.no_grad()
def eval_detector(model, ds, batch, workers, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, heat, reg, logwh, ind, mask in loader(ds, batch, False, workers, device):
        x, heat, reg, logwh, ind, mask = (t.to(device) for t in (x, heat, reg, logwh, ind, mask))
        loss, _ = detector_loss(model(x), heat, reg, logwh, ind, mask)
        tot += float(loss) * len(x)
        n += len(x)
    return tot / max(1, n)


# ---- re-ID ------------------------------------------------------------------------------

def train_reid(a) -> str:
    device = pick_device(a.device)
    root = root_of(a.data, a.root)
    tr_items, ev_items = load(a.data, "reid_train.json"), load(a.data, "reid_eval.json")
    ids = sorted({x["id"] for x in tr_items})
    if len(ids) < 2:
        raise SystemExit(f"re-ID needs at least 2 identities in {a.data} (found {len(ids)}): record more "
                         "people, or sessions with a --subject")
    run_dir = os.path.join(a.runs, a.name)
    os.makedirs(run_dir, exist_ok=True)
    cache = os.path.join(a.data, "reid_cache")
    print(f"re-ID: {len(tr_items)} crops / {len(ids)} identities (train), {len(ev_items)} eval crops, device {device}")

    model = ReIDNet(len(ids), width=a.width).to(device)
    if a.init:
        print(f"  backbone from {a.init}: {len(load_backbone(model, a.init))} tensors")
    ema = ModelEMA(model)
    tr = ReIDDataset(tr_items, root, cache, train=True, seed=a.seed)
    sampler = PKSampler(tr.labels, a.p, a.k, a.seed)
    tl = DataLoader(tr, batch_sampler=sampler, num_workers=a.workers, pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    total, step = a.epochs * len(sampler), 0
    ev = ReIDDataset(ev_items, root, cache, train=False) if ev_items else None
    best = -1.0

    for epoch in range(a.epochs):
        model.train()
        tr.epoch = epoch
        t0, sums, n = time.time(), {}, 0
        for x, y in tl:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, a.lr, warmup=min(500, total // 10 + 1))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            feat, logits = model(x)
            loss, parts = reid_loss(feat, logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1
            n += 1
            for k, v in {"loss": float(loss), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v
            if a.max_steps and step >= a.max_steps:
                break
        row = {"epoch": epoch, **{k: round(v / max(1, n), 4) for k, v in sums.items()},
               "s": round(time.time() - t0, 1)}
        score = -row["loss"]
        if ev is not None:
            r = eval_reid(ema.ema, ev, a.batch, a.workers, device)
            row.update({f"eval_{k}": v for k, v in r.items()})
            if r.get("map") == r.get("map"):          # not NaN
                score = r["map"]
        print(f"  epoch {epoch + 1}/{a.epochs} {row}")
        log_line(run_dir, row)
        extra = {"width": a.width, "num_ids": len(ids)}
        save_ckpt(os.path.join(run_dir, "last.pt"), ema.ema, epoch, extra)
        if score > best:
            best = score
            save_ckpt(os.path.join(run_dir, "best.pt"), ema.ema, epoch, extra)
        if a.max_steps and step >= a.max_steps:
            break

    net = ReIDNet(len(ids), width=a.width)
    net.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")["model"])
    wrapper = ReIDExport(net)
    onnx_path = os.path.join(run_dir, "reid_own.onnx")
    dummy = torch.randn(2, 3, 256, 128)
    export_onnx(wrapper, dummy, onnx_path, dynamic_batch=True)
    diff = check_with_cv2(wrapper, dummy, onnx_path)
    print(f"  exported {onnx_path}; cv2.dnn vs torch max diff {diff:.2e} {'OK' if diff < 1e-3 else 'CHECK THIS'}")
    write_card(run_dir, model="reid_own", kind="person re-ID (ours)", input="Nx3x256x128 RGB, ImageNet mean/std",
               output="Nx256 embedding (cosine)", width=a.width, epochs=a.epochs, lr=a.lr,
               p=a.p, k=a.k, identities=len(ids), device=str(device),
               init=a.init or "scratch (no pretrained weights)", data=a.data,
               dataset_manifest=load(a.data, "manifest.json"), best_score=best, cv2_max_diff=diff,
               usage=f"--reid onnx --reid-model {onnx_path}; set acquire/keep thresholds from tools/benchmark.py reid")
    return onnx_path


@torch.no_grad()
def eval_reid(model, ds, batch, workers, device) -> dict:
    from evaluation.metrics import reid_scores
    model.eval()
    embs = []
    for x, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
        embs.append(model.features(x.to(device))[1].cpu().numpy())
    if not embs:
        return {}
    emb = np.concatenate(embs)
    return reid_scores(emb, [i["id"] for i in ds.items], [i["group"] for i in ds.items],
                       [i["session"] for i in ds.items])


# ---- smoke test: the whole pipeline on synthetic data -----------------------------------------

def smoke(a) -> int:
    from dataset.export import export
    from dataset.synthetic import make_dataset
    from perception.body_reid import OnnxReidEmbedder
    from perception.person_detector import CenterPersonDetector
    import cv2
    work = a.work or tempfile.mkdtemp(prefix="df_smoke_")
    root, exp = os.path.join(work, "datasets"), os.path.join(work, "export")
    print(f"smoke: working in {work}")
    make_dataset(root, n_frames=24)
    m = export(root, exp)
    print(f"  export: {m['counts']}")
    common = dict(data=exp, root=None, runs=os.path.join(work, "runs"), device=a.device, workers=0,
                  seed=0, width=0.5, wd=0.01, init=None, max_steps=0)
    det = argparse.Namespace(**common, name="det", size=160, epochs=a.epochs, batch=8, lr=2e-3)
    det_onnx = train_detector(det)
    d = CenterPersonDetector(det_onnx, 160, conf=0.05)
    item = load(exp, "detector_eval.json")[0]
    found = d.detect(cv2.imread(os.path.join(root, item["image"])))
    print(f"  own detector via perception: {len(found)} boxes on an eval frame (truth: {len(item['boxes'])})")
    reid = argparse.Namespace(**{**common, "init": os.path.join(work, "runs", "det", "best.pt")},
                              name="reid", epochs=a.epochs, batch=16, lr=1e-3, p=4, k=4)
    reid_onnx = train_reid(reid)
    e = OnnxReidEmbedder(reid_onnx)
    ri = load(exp, "reid_eval.json")[:3]
    img = cv2.imread(os.path.join(root, ri[0]["image"]))
    v = e.embed(img, [tuple(ri[0]["box"])])
    print(f"  own re-ID via perception: embedding {v.shape}, norm {np.linalg.norm(v[0]):.3f}")
    ok = v.shape[1] == 256
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


# ---- CLI --------------------------------------------------------------------------------------

def parse(argv=None):
    p = argparse.ArgumentParser(description="train our own detector / re-ID models")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, epochs, batch, lr):
        sp.add_argument("--data", required=True, help="export folder from tools/data.py export")
        sp.add_argument("--root", help="dataset root (default: from the export's manifest)")
        sp.add_argument("--name", required=True, help="run name: output goes to runs/<name>/")
        sp.add_argument("--runs", default="runs")
        sp.add_argument("--epochs", type=int, default=epochs)
        sp.add_argument("--batch", type=int, default=batch)
        sp.add_argument("--lr", type=float, default=lr)
        sp.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay")
        sp.add_argument("--width", type=float, default=1.0, help="network width multiplier (0.5 = faster, weaker)")
        sp.add_argument("--init", help="start the backbone from this checkpoint of ours")
        sp.add_argument("--device", default="auto", help="auto / mps / cuda / cpu")
        sp.add_argument("--workers", type=int, default=4, help="data loading processes")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max-steps", type=int, default=0, help="stop early (debugging)")

    d = sub.add_parser("detector", help="our person detector")
    common(d, epochs=80, batch=32, lr=2e-3)
    d.add_argument("--size", type=int, default=320, help="square input size (multiple of 32)")

    r = sub.add_parser("reid", help="our person re-ID net")
    common(r, epochs=60, batch=64, lr=1e-3)
    r.add_argument("--p", type=int, default=16, help="identities per batch")
    r.add_argument("--k", type=int, default=4, help="crops per identity per batch")

    s = sub.add_parser("smoke", help="whole pipeline on synthetic data: proves this machine is set up")
    s.add_argument("--device", default="auto")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--work", help="keep the files here (default: a temp folder)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    if a.cmd == "detector":
        train_detector(a)
    elif a.cmd == "reid":
        train_reid(a)
    else:
        return smoke(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
