"""PyTorch side of our own models. Skipped where torch is not installed (CI,
the plain venv); run on the Mac after `./setup.sh --train`. The fuller check
is `python -m training.train smoke`."""

import os
import shutil
import tempfile
import unittest
from argparse import Namespace

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

import numpy as np


@unittest.skipUnless(HAVE_TORCH, "torch not installed (./setup.sh --train)")
class TestModels(unittest.TestCase):
    def test_detector_shapes_loss_and_input_norm_stays_frozen(self):
        from training.models import CenterNet, CenterNetExport, detector_loss
        from training.targets import encode_center
        net = CenterNet(width=0.5)
        net.train()
        self.assertFalse(net.norm.training)                  # never updates its statistics
        x = torch.rand(2, 3, 128, 128) * 255
        heat, reg, wh = net(x)
        self.assertEqual(tuple(heat.shape), (2, 1, 32, 32))
        t = [torch.from_numpy(np.stack(v)) for v in zip(*[encode_center([[10, 10, 60, 120]], 128)] * 2)]
        loss, parts = detector_loss((heat, reg, wh), *t)
        loss.backward()
        self.assertTrue(np.isfinite(float(loss)))
        self.assertIsNone(net.norm.weight.grad)
        out = CenterNetExport(net.eval())(x)
        self.assertEqual(tuple(out.shape), (2, 5, 32, 32))
        self.assertTrue(float(out[:, 0].min()) >= 0 and float(out[:, 0].max()) <= 1)

    def test_reid_fold_matches_bn_neck(self):
        from training.models import ReIDExport, ReIDNet
        net = ReIDNet(5, width=0.5)
        net.train()
        for _ in range(3):                                   # give the BN neck real statistics
            net(torch.randn(8, 3, 256, 128))
        net.eval()
        x = torch.randn(3, 3, 256, 128)
        with torch.no_grad():
            want = net.features(x)[1]
            got = ReIDExport(net)(x)
        self.assertTrue(torch.allclose(want, got, atol=1e-4))

    def test_triplet_prefers_separated_identities(self):
        from training.models import batch_hard_triplet
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
        bad = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0.1], [0.1, 1.0]])
        self.assertLess(float(batch_hard_triplet(good, labels)), float(batch_hard_triplet(bad, labels)))


@unittest.skipUnless(HAVE_TORCH, "torch not installed (./setup.sh --train)")
class TestSmoke(unittest.TestCase):
    """The whole pipeline, tiny: synthetic data -> export -> train both -> ONNX -> perception."""

    def test_smoke(self):
        from training.train import smoke
        work = tempfile.mkdtemp(prefix="df_smoke_")
        try:
            self.assertEqual(smoke(Namespace(device="cpu", epochs=1, work=work)), 0)
            self.assertTrue(os.path.exists(os.path.join(work, "runs", "det", "person_own.onnx")))
            self.assertTrue(os.path.exists(os.path.join(work, "runs", "reid", "model_card.json")))
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
