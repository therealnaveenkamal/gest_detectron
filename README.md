# Gest Detectron

**Gest Detectron** is a Python package designed to build an automatic pipeline for tracking hand movements in videos. It combines  **Google MediaPipe Hand Landmark Detection** and **Segment Anything Model (SAM2)** to deliver highly accurate gesture segmentation and tracking.

---

## Features

- **DinoV2**: When MediaPipe fails to detect hand landmarks due to occlusions, motion blur, or finger overlaps, we employ DinoV2 with a segmentation decoder head to detect hands and generate an initial segmentation mask. This allows us to overcome limitations in traditional landmark-based tracking.
- **SAM2 Integration**: We use the Segment Anything Model (SAM2) to refine hand segmentation. A few key points from the DinoV2-generated mask are extracted and used as prompts for SAM2 to generate high-resolution segmentation across the entire video.
- **Customizable Pipeline**: Provides flexibility for fine-tuning model parameters to improve segmentation accuracy.

---

## Why Dinov2 + Segmentation?

DinoV2, a Self-Supervised Learning (SSL) vision transformer, is used for detecting hands in cases where MediaPipe fails due to occlusions or motion blur. Its SSL-based feature extraction enables robust recognition of hand structures without labeled supervision, making it ideal for generalizing across diverse hand appearances and environments. The DinoV2 backbone was fine tuned on EgoHands dataset. By integrating a segmentation head, we extract a sparse set of random key points from the generated mask to feed as prompts for SAM2. These points serve as initialization cues, allowing SAM2 to propagate segmentation across the entire video.


#### Note: To avoid local installation and easier access, you may run the pipeline using 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YTSRAOQTTAQ_lTtdUsTwNPqC-yz7zL0O?usp=sharing)

## Installation
1. Setup
    ```bash
    conda create -n sam2 python=3.12
    conda activate sam2
    ```

2. Clone the repository:
    ```bash
    git clone -b dino_pipeline https://github.com/therealnaveenkamal/gest_detectron
    cd gest_detectron
    ```

2. Install the package:
    ```bash
    pip install .
    ```

3. Ensure PyTorch with CUDA is installed (for GPU support):
    ```bash
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
    ```

---

## Usage

### Detect and Segment Hands in a Video

In the gest_detectron directory, run the package from the command line to execute the SAM2 pipeline:

```bash
gest-detectron
```

## Package Enhancement and Fine-Tuning

This package can be easily modified to be used as a module and can be imported with input and output video parameters.
This pipeline supports fine-tuning at later stages for improved segmentation accuracy. Fine-tuning can enhance SAM2's ability to adapt to diverse lighting conditions, hand shapes, or motion speeds. Negative Clicks can help remove segments from certain frames.

---

## Output Examples

### Video Outputs
- **Segmented Video**: Displays masks applied to detected hands. Autosaved as `sam_masked_output_final.mp4` in the PWD.
 




https://github.com/user-attachments/assets/ec071927-b689-4db3-a4df-51a8597ce747






- **Landmarked Video**: NO HAND LANDMARKS DETECTED. Autosaved as `landmarked_output.mp4` in the PWD.




https://github.com/user-attachments/assets/f51601e7-17cc-42ef-a844-e08d338f343e





### Images

![detectron_image](https://github.com/user-attachments/assets/af764359-4519-4005-b77c-b51acfb056d3)

---

## Project Structure

- **`gest_detectron/`**: Contains the main package modules.
- **`requirements.txt`**: Lists required Python dependencies.
- **`setup.py`**: Configures the package for installation.
- **`GestDetectron_SAM_Notebook`**: Includes Jupyter Notebook for SAM2 pipeline.

---

## Technical Details

### DinoV2 Hand Segmentation
When MediaPipe fails to recognize hand landmarks due to occlusions or complex hand movements, we use DinoV2 with a segmentation head to detect hands and generate a segmentation mask. This approach ensures robust hand detection even in cases where traditional landmark-based methods struggle.

### SAM2: Segment Anything Model
SAM2 is a high-performance segmentation framework designed to adapt to diverse object shapes and contexts. It utilizes outputs from MediaPipe as input prompts (e.g., bounding boxes or click points) to generate segmentation masks. In this project, we used 5 TIPS and 1 WRIST point to create individual objects for both hands. The number of objects created and the landmarks used will depend on the segmentation evaluation criteria. Note that this SAM2-segmented video has not been evaluated with any ground truths.

![image](https://github.com/user-attachments/assets/144a15c7-863e-4986-9c62-7f41508c16fb)


---


## Author

**Naveenraj Kamalakannan**  
Email: [naveenraj.k@nyu.edu](mailto:naveenraj.k@nyu.edu)  
GitHub: [https://github.com/therealnaveenkamal/gest_detectron](https://github.com/therealnaveenkamal/gest_detectron)  
Portfolio: [Website](https://itsnav.com/)

---

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Acknowledgments

This project leverages:
- [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Segment Anything Model (SAM)](https://segment-anything.com/)

Special thanks to Professor [Carlos Fernandez-Granda](https://math.nyu.edu/~cfgranda/)
