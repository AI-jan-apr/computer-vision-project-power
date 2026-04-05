[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nSn4fJNC)
# Computer Vision Project

## 📌 Project Overview

This project aims to build a Computer Vision system for smart parking management. The system detects cars in parking areas and determines whether each parking space is occupied or available. In addition, it identifies vehicles that are not parked correctly, which can be used to issue parking violations. The system also tracks how long each vehicle remains parked in a specific space.

The project applies deep learning and object detection techniques to solve a real-world problem in parking monitoring and management. It demonstrates the use of computer vision for vehicle detection, parking occupancy analysis, parking violation detection, and parking duration tracking.

---

## 👥 Team Information

| Name | Role | Contribution |
|------|------|-------------|
| Abdulslam | Team Leader | Project management, model training, model improvement, evaluation, and data labeling |
| Faisal | Model Development | Model training, model improvement, and data labeling |
| Hanan | ROI & Data Collection | ROI implementation, data collection, and data labeling |
| Fozia | Data Collection & Presentation | Data collection, data labeling, and presentation preparation |

---

## 🎯 Project Objectives

- Develop a Computer Vision system to detect cars in parking areas using YOLO  
- Determine whether each parking space is occupied or available  
- Identify incorrectly parked vehicles to support parking violation detection  
- Track the parking duration for each vehicle  
- Improve parking monitoring and management using an automated intelligent system  

### Problem Definition
Parking management is often inefficient and requires manual monitoring. There is a need for an automated system that can accurately detect vehicles, monitor parking occupancy, and identify violations.

### Importance of the Problem
Efficient parking management helps reduce congestion, improves space utilization, and supports better enforcement of parking rules.

### Expected Outcome
A working system capable of detecting cars, identifying parking status (occupied or empty), detecting violations, and estimating parking time with acceptable performance.
---

## 📂 Dataset

- A combination of two datasets was used:
  - **COCO Dataset** as a pretrained base, which includes the "car" class  
  - **PKLot Dataset** for parking environments, containing top-view images  

### Data Source
- COCO Dataset (general dataset for pretraining)  
- PKLot Dataset (specialized dataset for parking scenarios)  
- **Roboflow** was used for data labeling and management  

### Dataset Description
- The dataset includes:
  - Images of parking areas  
  - Bounding box annotations for **cars (car)**  
  - Bounding box annotations for **parking slots**  

### Number of Samples
- The dataset consists of labeled images  
- You can add the exact number if available  

### Preprocessing
- Converted data into YOLO format (images + labels)  
- Standardized class labels (car and parking)  
- Organized dataset structure (dataset/images and dataset/labels)  
- Used Roboflow for data cleaning and annotation verification  

### Notes
- The dataset was highly biased toward top-view vehicles, which affected the model’s ability to generalize to other perspectives such as side-view  
- The availability of parking slot labels helped in determining whether a parking space is occupied or empty  

---

## 🧠 Methodology

### 1. Data Preprocessing
- The dataset was prepared and organized in YOLO format, including images and corresponding label files  
- Bounding box annotations were created for both **cars (car)** and **parking slots** using Roboflow  
- The dataset was cleaned and structured to ensure compatibility with YOLO training and evaluation  
- The data mainly focused on parking-lot scenes, especially top-view images  

### 2. Model Selection
- **YOLO** was selected as the main object detection model due to its speed and efficiency in real-time detection tasks  
- A pretrained YOLO model was used as a starting point to leverage general knowledge learned from the COCO dataset  
- YOLO was chosen because it can detect multiple objects in the same image, such as cars and parking slots  

### 3. Training Process
- The YOLO model was fine-tuned using the project dataset  
- Training was performed on labeled parking-lot images containing cars and parking-slot annotations  
- The model learned to detect vehicles and identify parking-slot locations  
- The training process aimed to improve detection performance in parking environments  

### 4. Evaluation Method
- The model was evaluated using standard object detection metrics:
  - **Precision**
  - **Recall**
  - **mAP@50**
  - **mAP@50-95**
- These metrics were used to measure detection accuracy and generalization ability  
- The evaluation results helped identify model limitations, especially in handling different viewpoints such as side-view vehicles   

---

## ⚙️ Implementation

### Tools and Libraries
- Python  
- Ultralytics YOLO  
- Roboflow (for dataset labeling and management)  
- Jupyter Notebook  

### Key Implementation Steps
- Loaded a pretrained YOLO model  
- Prepared the dataset in YOLO format (images and labels)  
- Trained the model to detect **cars** and **parking slots**  
- Evaluated the model using validation data  
- Used **IoU (Intersection over Union)** to:
  - Measure the overlap between predicted and ground truth bounding boxes  
  - Determine detection accuracy  

- Used detection results to:
  - Identify whether parking spaces are occupied or empty  
  - Detect incorrectly parked vehicles  
  - Estimate parking duration  

### Challenges and Solutions
- **Dataset bias (top-view only)**  
  - The dataset initially contained mostly top-view vehicles, which affected the model’s ability to generalize  

- **Low Recall (missed detections)**  
  - The initial model was trained mainly on top-view data, making it less capable of detecting vehicles from different angles such as side-view  
  - To improve performance, a new model was trained using more diverse data (top-view and side-view)  
  - Fine-tuning was applied on the YOLO model, which improved generalization and increased the number of detected vehicles  

- **Parking slot detection using ROI (Region of Interest)**  
  - ROI was used to define parking areas and determine whether they are occupied or empty  
  - However, it requires manual setup for each parking space, which limits scalability  

- **Handling multiple tasks (cars + parking slots)**  
  - This was addressed by labeling both classes and training the model to detect them within the same system  
---

## 📊 Results

### Model Performance

- **Old Model**  
  Precision: 76.8% | Recall: 62.8% | mAP@50: 68.8%

- **New Model**  
  Precision: 70.8% | Recall: 79.7% | mAP@50: 73.2%

---

### Analysis

- The new model achieved higher **Recall**, meaning fewer missed cars  
- **mAP@50 improved**, indicating better overall performance  
- Slight drop in Precision is expected due to increased detection sensitivity  

---

### Conclusion

The improved model performs better in real-world scenarios by detecting more vehicles and handling different viewpoints more effectively.

---

## 🚀 How to Run the Project
Provide clear instructions:

```bash
# Example
pip install -r requirements.txt
python main.py