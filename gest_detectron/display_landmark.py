import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure video processing options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,  # VIDEO mode
    num_hands=2)

def process_video(input_path, output_path):
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_data = []

    with HandLandmarker.create_from_options(options) as detector:
        frame_timestamp = 0
        count = 0
        print("Hand Landmarking In Progress...")

        output_dir = "frames"
        os.makedirs(output_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Hand Landmarked Video Rendering Completed")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hand landmarks (with timestamp)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp)

            frame_entry = {
                "timestamp": frame_timestamp,
                "landmarks": [],
                "handedness": []
            }

            for hand_landmarks in detection_result.hand_landmarks:
                frame_entry["landmarks"].append([(lm.x, lm.y) for lm in hand_landmarks])

            for classification in detection_result.handedness:
                frame_entry["handedness"].append([(c.category_name, c.score) for c in classification])

            frame_data.append(frame_entry)

            # Draw landmarks (using your existing function)
            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

            # Convert back to BGR for video output
            bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Write processed frame
            out.write(bgr_frame)

            output_file = os.path.join(output_dir, f"{count:05d}.jpg")
            # Save the frame as a JPG with specified quality
            cv2.imwrite(output_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Increment frame timestamp on milliseconds
            frame_timestamp += int(1000 / fps)
            count+=1

    cap.release()
    out.release()
    return frame_data