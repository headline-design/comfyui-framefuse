"""ComfyUI node for stitching still frames onto IMAGE video batches."""

from collections.abc import Mapping


def _as_nhwc_batch(images, input_name: str):
    import torch

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{input_name} must be a torch.Tensor IMAGE batch.")
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4:
        raise ValueError(f"{input_name} must have shape [B,H,W,C] or [H,W,C].")
    if images.shape[0] < 1:
        raise ValueError(f"{input_name} must contain at least one frame.")
    if images.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{input_name} must use channel-last IMAGE layout [B,H,W,C].")
    return images


def _resize_nhwc_batch(images, height: int, width: int):
    import torch.nn.functional as F

    if images.shape[1] == height and images.shape[2] == width:
        return images

    original_dtype = images.dtype
    if not images.is_floating_point():
        images = images.float()

    resized = F.interpolate(
        images.movedim(-1, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    return resized.to(dtype=original_dtype).clamp(0, 1)


def stitch_frame_batch(
    video_frames,
    frame,
    frame_index: int,
    repeat_count: int,
    resize_frame_to_video: bool,
    placement_mode: str,
):
    import torch

    video_frames = _as_nhwc_batch(video_frames, "video_frames")
    frame_batch = _as_nhwc_batch(frame, "frame").to(device=video_frames.device, dtype=video_frames.dtype)

    target_height = int(video_frames.shape[1])
    target_width = int(video_frames.shape[2])
    if frame_batch.shape[1] != target_height or frame_batch.shape[2] != target_width:
        if not resize_frame_to_video:
            raise ValueError(
                "frame size does not match video_frames. Enable resize_frame_to_video or provide a matching frame."
            )
        frame_batch = _resize_nhwc_batch(frame_batch, target_height, target_width)

    selected_index = int(frame_index)
    if selected_index < 0:
        selected_index = int(frame_batch.shape[0]) + selected_index
    if selected_index < 0 or selected_index >= int(frame_batch.shape[0]):
        raise ValueError(f"frame_index {frame_index} is out of range for {int(frame_batch.shape[0])} frame(s).")

    repeats = max(1, int(repeat_count))
    selected_frame = frame_batch[selected_index:selected_index + 1]
    stitched_frames = selected_frame.repeat(repeats, 1, 1, 1)
    if placement_mode == "prepend_start":
        stitched_video = torch.cat((stitched_frames, video_frames), dim=0)
    else:
        stitched_video = torch.cat((video_frames, stitched_frames), dim=0)
    return stitched_video, selected_index, repeats


def stitch_audio_silence(
    audio,
    stitched_frame_count: int,
    fps: float,
    extend_audio_with_silence: bool,
    placement_mode: str,
):
    import torch

    if audio is None:
        return None, "No audio input."
    if not extend_audio_with_silence:
        return audio, "Audio passed through unchanged."
    if not isinstance(audio, Mapping):
        raise TypeError("extend_audio_with_silence is enabled, but audio was not a ComfyUI AUDIO mapping.")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or not sample_rate:
        raise ValueError(
            "extend_audio_with_silence is enabled, but audio did not contain a valid waveform tensor and sample_rate."
        )

    fps_value = float(fps)
    if fps_value <= 0:
        raise ValueError("fps must be greater than 0 when extending audio with silence.")

    silence_samples = int(round(float(sample_rate) * int(stitched_frame_count) / fps_value))
    if silence_samples <= 0:
        return audio, "Audio passed through unchanged because stitched silence duration rounded to 0 samples."

    silence_shape = list(waveform.shape)
    silence_shape[-1] = silence_samples
    silence = torch.zeros(silence_shape, dtype=waveform.dtype, device=waveform.device)
    silence_seconds = silence_samples / float(sample_rate)

    extended_audio = dict(audio)
    if placement_mode == "prepend_start":
        extended_audio["waveform"] = torch.cat((silence, waveform), dim=-1)
        return extended_audio, f"Prepended {silence_samples} silent audio sample(s) ({silence_seconds:.6f}s)."

    extended_audio["waveform"] = torch.cat((waveform, silence), dim=-1)
    return extended_audio, f"Appended {silence_samples} silent audio sample(s) ({silence_seconds:.6f}s)."


def trim_frame_batch_end(video_frames, trim_count: int):
    video_frames = _as_nhwc_batch(video_frames, "video_frames")

    trim_frames = max(0, int(trim_count))
    total_frames = int(video_frames.shape[0])
    if trim_frames == 0:
        return video_frames, trim_frames
    if trim_frames >= total_frames:
        raise ValueError(
            f"trim_count {trim_frames} would remove all {total_frames} frame(s). "
            "Leave at least one frame in the output batch."
        )

    return video_frames[:-trim_frames], trim_frames


def trim_audio_end(audio, trimmed_frame_count: int, fps: float, trim_audio: bool):
    import torch

    if audio is None:
        return None, "No audio input."
    if not trim_audio:
        return audio, "Audio passed through unchanged."
    if not isinstance(audio, Mapping):
        raise TypeError("trim_audio is enabled, but audio was not a ComfyUI AUDIO mapping.")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or not sample_rate:
        raise ValueError("trim_audio is enabled, but audio did not contain a valid waveform tensor and sample_rate.")

    fps_value = float(fps)
    if fps_value <= 0:
        raise ValueError("fps must be greater than 0 when trimming audio.")

    trim_samples = int(round(float(sample_rate) * int(trimmed_frame_count) / fps_value))
    if trim_samples <= 0:
        return audio, "Audio passed through unchanged because trimmed duration rounded to 0 samples."

    total_samples = int(waveform.shape[-1])
    kept_samples = max(0, total_samples - trim_samples)
    trimmed_audio = dict(audio)
    trimmed_audio["waveform"] = waveform[..., :kept_samples]
    trimmed_seconds = min(trim_samples, total_samples) / float(sample_rate)

    if trim_samples > total_samples:
        return (
            trimmed_audio,
            f"Trimmed all available audio ({trimmed_seconds:.6f}s requested exceeded waveform length).",
        )

    return trimmed_audio, f"Trimmed {trim_samples} audio sample(s) from the end ({trimmed_seconds:.6f}s)."


class FrameFuse:
    """Add a selected frame to the start or end of a ComfyUI IMAGE video batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video represented as a ComfyUI IMAGE batch, such as VHS_LoadVideo.IMAGE.",
                }),
                "frame": ("IMAGE", {
                    "tooltip": "Still frame or IMAGE batch. The selected frame is stitched onto the video batch.",
                }),
                "frame_index": ("INT", {
                    "default": -1,
                    "min": -100000,
                    "max": 100000,
                    "tooltip": "Frame index to use from frame. Use -1 for the last frame in that batch.",
                }),
                "repeat_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1024,
                    "tooltip": "How many copies of the selected frame to add.",
                }),
                "placement_mode": (["append_end", "prepend_start"], {
                    "default": "append_end",
                    "tooltip": "Add the selected frame(s) to the end or the beginning of the batch.",
                }),
                "resize_frame_to_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Resize frame to match video_frames before stitching.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 0.01,
                    "tooltip": "Frame rate used when extending audio with silence.",
                }),
                "extend_audio_with_silence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add matching silence to the audio in the same direction as the stitched frames.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio to pass through or extend with silence.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("STITCHED_VIDEO", "AUDIO", "FRAME_COUNT", "REPORT")
    FUNCTION = "stitch"
    CATEGORY = "FrameFuse"

    def stitch(
        self,
        video_frames,
        frame,
        frame_index: int,
        repeat_count: int,
        placement_mode: str,
        resize_frame_to_video: bool,
        fps: float,
        extend_audio_with_silence: bool,
        audio=None,
    ):
        stitched_video, selected_index, repeats = stitch_frame_batch(
            video_frames,
            frame,
            frame_index,
            repeat_count,
            resize_frame_to_video,
            placement_mode,
        )
        output_audio, audio_report = stitch_audio_silence(
            audio,
            repeats,
            fps,
            extend_audio_with_silence,
            placement_mode,
        )
        action = "Prepended" if placement_mode == "prepend_start" else "Appended"
        report = (
            f"{action} {repeats} frame(s) from frame index {selected_index}. "
            f"Output frame count: {int(stitched_video.shape[0])}. "
            f"{audio_report}"
        )
        return (stitched_video, output_audio, int(stitched_video.shape[0]), report)


class FrameFuseTrimEnd:
    """Trim frames from the end of a ComfyUI IMAGE video batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video represented as a ComfyUI IMAGE batch.",
                }),
                "trim_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1024,
                    "tooltip": "How many frames to remove from the end of the batch.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 0.01,
                    "tooltip": "Frame rate used to calculate matching audio trim duration.",
                }),
                "trim_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Trim matching audio duration from the end of the audio input.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio to pass through or trim from the end.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("TRIMMED_VIDEO", "AUDIO", "FRAME_COUNT", "REPORT")
    FUNCTION = "trim"
    CATEGORY = "FrameFuse"

    def trim(
        self,
        video_frames,
        trim_count: int,
        fps: float,
        trim_audio: bool,
        audio=None,
    ):
        trimmed_video, trimmed_frames = trim_frame_batch_end(video_frames, trim_count)
        output_audio, audio_report = trim_audio_end(audio, trimmed_frames, fps, trim_audio)
        report = (
            f"Trimmed {trimmed_frames} frame(s) from the end. "
            f"Output frame count: {int(trimmed_video.shape[0])}. "
            f"{audio_report}"
        )
        return (trimmed_video, output_audio, int(trimmed_video.shape[0]), report)


NODE_CLASS_MAPPINGS = {
    "FrameFuse": FrameFuse,
    "FrameFuseTrimEnd": FrameFuseTrimEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameFuse": "FrameFuse",
    "FrameFuseTrimEnd": "FrameFuse Trim End",
}
