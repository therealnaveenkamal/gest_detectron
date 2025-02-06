from sam2.build_sam import build_sam2_video_predictor
import torch
import torchvision
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2


def init_sam_predictor(model, device, sam2_checkpoint = "sam2.1_hiera_large.pt", model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", video_dir = "./frames"):
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device)
    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)
    prompts={}

    first_frame_path = os.path.join(video_dir, "00000.jpg")
    _, mask_np = predict_mask(model, first_frame_path)


    y, x = np.where(np.array(mask_np) > 0.8 * 255)

    num_points = 10

    if len(x) > 0:
        sampled_idx = np.random.choice(len(x), min(num_points, len(x)), replace=False)
        points = np.column_stack([x[sampled_idx], y[sampled_idx]]).astype(np.float32)
        labels = np.ones(len(points), dtype=np.int32)
    else:
        points = np.empty((0, 2), dtype=np.float32)
        labels = np.empty(0, dtype=np.int32)


    ann_frame_idx = 0
    ann_obj_id = 1
    prompts[ann_obj_id] = points, labels


    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    return predictor, inference_state
