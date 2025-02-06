from gest_detectron.display_landmark import process_video
from gest_detectron.initialize_sam2 import show_mask, show_points, show_box
from gest_detectron.initialize_mask import init_sam_predictor
from gest_detectron.render_sam import render_sam_video

import torch
import torchvision
import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import torch.nn as nn
from torchvision.models import resnet50
import torch.hub

import torchvision.transforms as T


class DinoV2Segmentation(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16->32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32->64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64->128
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True),  # 128->224
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Get features (B, 256, 768)
        features = self.dinov2.forward_features(x)['x_norm_patchtokens']

        # Reshape to (B, 768, 16, 16)
        features = features.permute(0, 2, 1).view(-1, 768, 16, 16)

        return self.decoder(features)

def main():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Generating Hand Landmarks
    frame_data = process_video("final_corrected.mp4", "landmarked_output.mp4")

    if(len(frame_data[0]['landmarks'])==0):
        print("NO MEDIAPIPE HANDLANDMARKS WERE DETECTED. LANDMARK PIPELINE FAILURE! DINOV2 CALLING")
    else:
        print("MEDIAPIPE HANDLANDMARKS DETECTED")

  
    print("Video Frame Extracted")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    

    print("Predictor Calling")

    model = DinoV2Segmentation(num_classes=1)
    model.load_state_dict(torch.load("models/dinov2_hand_trained_segmentation.pth", map_location='cpu'))
    model.eval()

    predictor, inference_state = init_sam_predictor(
        model, 
        device, 
        sam2_checkpoint="models/sam2.1_hiera_large.pt", 
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", 
        video_dir="frames"
    )
    print("Predictor Initialized")

    # Video processing
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Render segmented video
    render_sam_video(
        video_path="final_corrected.mp4",
        video_segments=video_segments,
        output_path="sam_masked_output_final.mp4",
        alpha=1
    )
    print("Video rendering complete.")

if __name__ == "__main__":
    main()
