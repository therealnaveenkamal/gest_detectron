from gest_detectron.display_landmark import process_video
from gest_detectron.initialize_sam2 import show_mask, show_points, show_box
from gest_detectron.initialize_mask import init_sam_predictor
from gest_detectron.render_sam import render_sam_video


import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from sam2.build_sam import build_sam2_video_predictor

# Generating Hand Landmarks 

frame_data = process_video("test.mp4", "landmarked_output.mp4")

print("Video Frame Extracted")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


# if device.type == "cuda":
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )

print("Predictor Calling")

predictor, inference_state = init_sam_predictor(frame_data, device, sam2_checkpoint = "sam2.1_hiera_large.pt", model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", video_dir = "./frames")

print("Predictor Initialized")

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


render_sam_video(
    video_path="test.mp4",
    video_segments=video_segments,
    output_path="sam_masked_output_final.mp4",
    alpha=1
)
