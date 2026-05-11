# AI-Based IVF Embryo Grading System

Manual embryo grading is subjective and varies across labs and specialists. I built a computer vision system using YOLOv8 segmentation that automatically calculates fragmentation percentage from microscope images and assigns Grade A, B, or C to each embryo.

**mAP@50: 76.6% | Inference Speed: 127ms**

**Tech Stack:** Python, YOLOv8, Roboflow, OpenCV, Streamlit

**Methodology:** Followed CRISP-ML(Q) framework covering data collection, annotation, model training, evaluation, and Streamlit deployment.

`preprocess_before_annotation.py` handles image cleaning and resizing before annotation.

`Embryo_YOLOv8_Training.ipynb` is the training notebook built on Google Colab.

![Architecture](Untitled%20Diagram.drawio.png)
