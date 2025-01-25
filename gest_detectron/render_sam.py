import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from sam2.build_sam import build_sam2_video_predictor


def render_sam_video(video_path, video_segments, output_path, alpha=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create color mapping for objects
    obj_ids = list({k for frame in video_segments.values() for k in frame.keys()})

    frame_idx = 0

    output_dir = "segments"
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in video_segments:
            # Convert to RGB for processing (SAM masks are in RGB space)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame_rgb.copy()

            for obj_id, mask in video_segments[frame_idx].items():
                # Ensure mask is 2D and matches frame dimensions
                mask = mask[0]
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height))

                cmap = plt.get_cmap("tab10")

                if(obj_id <=20):
                  color = [255, 0, 0]
                else:
                  color = [0, 255, 0]

                # Create colored mask
                mask_bgr = np.zeros_like(overlay)
                mask_bgr[mask] = color

                # Blend mask with overlay
                overlay = cv2.addWeighted(overlay, 1, mask_bgr, alpha, 0)

            # Convert back to BGR for video writing
            frame_out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            output_file = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
            # Save the frame as a JPG with specified quality
            cv2.imwrite(output_file, frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            out.write(frame_out)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved SAM masked video to {output_path}")