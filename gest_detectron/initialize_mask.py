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



def init_sam_predictor(fd, device, sam2_checkpoint = "sam2.1_hiera_large.pt", model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", video_dir = "./frames"):
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cpu")
    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)
    prompts={}

    cap = cv2.VideoCapture("test.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    ann_frame_idx = 0  # the frame index we interact with
    ic, ic2 = 0, 0

    for ic, elem in enumerate(fd[0]['landmarks'][0]):
        if(ic%4 ==0):
            temp = []
            ann_obj_id = ic
            temp.append([elem[0]*width, elem[1]*height])
            points = np.array(temp, dtype=np.float32)
            labels = np.array(np.ones(1), np.int32)
            
            prompts[ann_obj_id] = points, labels
            
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

    ic+=1
    for ic2, elem in enumerate(fd[0]['landmarks'][1]):
        if(ic2%4 ==0):
            temp = []
            ann_obj_id = ic+ic2
            temp.append([elem[0]*width, elem[1]*height])
            points = np.array(temp, dtype=np.float32)
            labels = np.array(np.ones(1), np.int32)
            
            prompts[ann_obj_id] = points, labels
            
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

    return predictor, inference_state
