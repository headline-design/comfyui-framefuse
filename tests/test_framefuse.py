import importlib.util
import sys
import unittest
from collections.abc import Mapping
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
    class LazyAudioMap(Mapping):
        def __init__(self, payload):
            self.payload = payload

        def __getitem__(self, key):
            return self.payload[key]

        def __iter__(self):
            return iter(self.payload)

        def __len__(self):
            return len(self.payload)

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

    def test_stitch_audio_silence_accepts_mapping_audio(self):
        audio = self.LazyAudioMap({
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        })

        extended, report = framefuse_nodes.stitch_audio_silence(audio, 10, 30.0, True, "prepend_start")

        self.assertEqual(extended["waveform"].shape, (1, 2, 64000))
        self.assertTrue(torch.equal(extended["waveform"][..., :16000], torch.zeros((1, 2, 16000))))
        self.assertIn("0.333333", report)

    def test_stitch_audio_silence_rejects_non_dict_audio_when_extension_enabled(self):
        with self.assertRaises(TypeError):
            framefuse_nodes.stitch_audio_silence("not-audio", 2, 24.0, True, "prepend_start")

    def test_stitch_audio_silence_rejects_invalid_audio_dict_when_extension_enabled(self):
        with self.assertRaises(ValueError):
            framefuse_nodes.stitch_audio_silence({"waveform": None, "sample_rate": None}, 2, 24.0, True, "append_end")

    def test_trim_frame_batch_end_removes_tail_frames(self):
        video = torch.arange(4 * 2 * 2 * 3, dtype=torch.float32).reshape(4, 2, 2, 3)

        trimmed, trimmed_count = framefuse_nodes.trim_frame_batch_end(video, 2)

        self.assertEqual(trimmed.shape, (2, 2, 2, 3))
        self.assertEqual(trimmed_count, 2)
        self.assertTrue(torch.equal(trimmed, video[:2]))

    def test_trim_frame_batch_end_rejects_removing_all_frames(self):
        video = torch.zeros((2, 4, 4, 3))

        with self.assertRaises(ValueError):
            framefuse_nodes.trim_frame_batch_end(video, 2)

    def test_trim_audio_end_removes_matching_tail_audio(self):
        audio = {
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        }

        trimmed, report = framefuse_nodes.trim_audio_end(audio, 2, 24.0, True)

        self.assertEqual(trimmed["waveform"].shape, (1, 2, 44000))
        self.assertTrue(torch.equal(trimmed["waveform"], audio["waveform"][..., :44000]))
        self.assertIn("Trimmed 4000 audio sample(s)", report)

    def test_trim_audio_end_passes_through_when_disabled(self):
        audio = {
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        }

        trimmed, report = framefuse_nodes.trim_audio_end(audio, 10, 30.0, False)

        self.assertIs(trimmed, audio)
        self.assertIn("passed through unchanged", report)

    def test_trim_audio_end_accepts_mapping_audio(self):
        audio = self.LazyAudioMap({
            "waveform": torch.ones((1, 2, 48000)),
            "sample_rate": 48000,
        })

        trimmed, report = framefuse_nodes.trim_audio_end(audio, 10, 30.0, True)

        self.assertEqual(trimmed["waveform"].shape, (1, 2, 32000))
        self.assertIn("0.333333", report)


if __name__ == "__main__":
    unittest.main()
