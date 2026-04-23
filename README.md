# FrameFuse

Standalone ComfyUI custom node for stitching one selected `IMAGE` frame onto the beginning or end of an `IMAGE` video batch, with optional matching silence added to an `AUDIO` stream.

## What it does

`FrameFuse` is a small utility node for cases where you already have a video as an `IMAGE` batch in ComfyUI and want to:

- hold on a still frame at the end of the clip
- add a still frame at the start of the clip
- repeat that frame multiple times
- keep the audio aligned by prepending or appending silence

This node works on ComfyUI `IMAGE` batches. It does not mux or rewrite video container files directly.

## Node

- `FrameFuse`

Inputs:

- `video_frames`: source `IMAGE` batch
- `frame`: still frame or `IMAGE` batch to sample from
- `frame_index`: which frame to use from `frame`
- `repeat_count`: how many copies of that frame to stitch in
- `placement_mode`: `append_end` or `prepend_start`
- `resize_frame_to_video`: resize the frame to match the video size
- `fps`: used to calculate matching silence duration
- `extend_audio_with_silence`: whether to insert silence into the audio
- `audio` (optional): ComfyUI `AUDIO` input

Outputs:

- `STITCHED_VIDEO`
- `AUDIO`
- `FRAME_COUNT`
- `REPORT`

## Typical wiring

```text
VHS_LoadVideo.IMAGE  -> FrameFuse.video_frames
Still IMAGE          -> FrameFuse.frame
VHS_LoadVideo.audio  -> FrameFuse.audio

FrameFuse.STITCHED_VIDEO -> VHS_VideoCombine.images
FrameFuse.AUDIO          -> VHS_VideoCombine.audio
```

## Install locally

Clone or copy this folder into your ComfyUI `custom_nodes/` directory:

```text
custom_nodes/comfyui-framefuse
```

Restart ComfyUI and add `FrameFuse` from the node search.

Official docs:

- https://docs.comfy.org/registry/publishing
- https://docs.comfy.org/registry/specifications
- https://docs.comfy.org/registry/standards
