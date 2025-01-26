# Gest Detectron

**Gest Detectron** is a Python package designed to build an automatic pipeline for tracking hand movements in videos. It combines  **Google MediaPipe Hand Landmark Detection** and **Segment Anything Model (SAM2)** to deliver highly accurate gesture segmentation and tracking.

---

## Features

- **Google MediaPipe Integration**: Utilizes the Hand Landmark Detection framework for precise detection of hand positions and movements.
- **SAM2 Integration**: Incorporates the Segment Anything Model (SAM2) for high-resolution segmentation, adaptable to various video and frame resolutions.
- **Customizable Pipeline**: Provides flexibility for fine-tuning model parameters to improve segmentation accuracy.

---

#### Note: To avoid local installation and easier access, you may run the pipeline using 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hQw75uHQ-DrgexBqsciD6JDDrk_TRz9K?usp=sharing)

## Installation
1. Setup
    ```bash
    conda create -n sam2 python=3.12
    conda activate sam2
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/therealnaveenkamal/gest_detectron
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

This package can we easily modified to be used as a module and can be imported with input and output video parameters.
This pipeline supports fine-tuning at later stages for improved segmentation accuracy. Fine-tuning can enhance SAM2's ability to adapt to diverse lighting conditions, hand shapes, or motion speeds. Negative Clicks can help remove segments from certain frames.

---

## Output Examples

### Video Outputs
- **Segmented Video**: Displays masks applied to detected hands. Autosaved as `sam_masked_output_final.mp4` in the PWD.
 




https://github.com/user-attachments/assets/ec071927-b689-4db3-a4df-51a8597ce747






- **Landmarked Video**: Shows detected hand landmarks superimposed on the video frames. Autosaved as `landmarked_output.mp4` in the PWD.




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

### MediaPipe Hand Landmark Detection
MediaPipe Hand Landmark Detection is a real-time framework for detecting hand key points. It outputs 21 3D landmarks for each hand, providing precise localization for hand gestures and motions.   

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
