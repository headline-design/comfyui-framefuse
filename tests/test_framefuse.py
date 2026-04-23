import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "nodes.py"
SPEC = importlib.util.spec_from_file_location("framefuse_nodes", NODES_PATH)
framefuse_nodes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = framefuse_nodes
SPEC.loader.exec_module(framefuse_nodes)


@unittest.skipIf(torch is None, f"torch is unavailable: {TORCH_IMPORT_ERROR}")
class FrameFuseTests(unittest.TestCase):
    def test_stitch_frame_batch_appends_frame(self):
        video = torch.zeros((2, 4, 4, 3))
        frame = torch.ones((1, 4, 4, 3))

        stitched, selected_index, repeats = framefuse_nodes.stitch_frame_batch(
            video, frame, -1, 1, True, "append_end"
        )

        self.assertEqual(stitched.shape, (3, 4, 4, 3))
        self.assertEqual(selected_index, 0)
        self.assertEqual(repeats, 1)
        self.assertTrue(torch.equal(stitched[-1], frame[0]))

    def test_stitch_frame_batch_prepends_frame(self):
        video = torch.zeros((2, 4, 4, 3))
        frame = torch.ones((1, 4, 4, 3))

        stitched, selected_index, repeats = framefuse_nodes.stitch_frame_batch(
            video, frame, 0, 2, True, "prepend_start"
        )

        self.assertEqual(stitched.shape, (4, 4, 4, 3))
        self.assertEqual(selected_index, 0)
        self.assertEqual(repeats, 2)
        self.assertTrue(torch.equal(stitched[0], frame[0]))
        self.assertTrue(torch.equal(stitched[1], frame[0]))
        self.assertTrue(torch.equal(stitched[2], video[0]))

    def test_stitch_audio_silence_appends_silence(self):
        audio = {
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        }

        extended, report = framefuse_nodes.stitch_audio_silence(audio, 2, 24.0, True, "append_end")

        self.assertEqual(extended["waveform"].shape, (1, 2, 52000))
        self.assertTrue(torch.equal(extended["waveform"][..., :48000], audio["waveform"]))
        self.assertTrue(torch.equal(extended["waveform"][..., 48000:], torch.zeros((1, 2, 4000))))
        self.assertIn("Appended", report)

    def test_stitch_audio_silence_prepends_silence(self):
        audio = {
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        }

        extended, report = framefuse_nodes.stitch_audio_silence(audio, 2, 24.0, True, "prepend_start")

        self.assertEqual(extended["waveform"].shape, (1, 2, 52000))
        self.assertTrue(torch.equal(extended["waveform"][..., :4000], torch.zeros((1, 2, 4000))))
        self.assertTrue(torch.equal(extended["waveform"][..., 4000:], audio["waveform"]))
        self.assertIn("Prepended", report)


if __name__ == "__main__":
    unittest.main()
