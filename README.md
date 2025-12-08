# **Mental Health Detection Using Deep Learning and Facial Action Units**

### **A Clinical Video-Based System for Depression Level Classification**

*(Using PYFEAT Action Units + ViT Fine-Tuned Model)*


## **Overview**

This project is a deep learning–based **mental health detection system** that analyzes facial expressions from video recordings to classify different levels of depression.
A clinical dataset was collected from **BVH (Bahawal Victoria Hospital)** consisting of:

* Normal users
* Patients with mild depression
* Patients with moderate depression
* Patients with severe depression

Each video is analyzed to extract **Facial Action Units (AUs)** using the **PyFEAT** toolkit, and then multiple deep learning models were evaluated for depression-level classification.

The system outputs a **depression category** and assigns a **score** based on intensity patterns of extracted AUs.


## **Key Features**

* Real clinical dataset collected from **BVH**
* Extraction of **Facial Action Units** using PyFEAT
* Automated scoring system based on AU intensity
* Classification of depression into:

  * Normal
  * Mild
  * Moderate
  * Severe
* Evaluation of multiple deep learning architectures
* Best performance achieved with **ViT (Vision Transformer)**
* Fine-tuned transformer-based workflow for high accuracy


## **How It Works**

1. Video of the user is processed frame-by-frame
2. **PyFEAT** extracts:

   * Facial landmarks
   * Action Units
   * AU intensities & occurrences
3. AU sequences are aggregated into temporal features
4. Deep learning models classify depression into four categories
5. Scoring system is applied to interpret AU intensities into a meaningful mental health score


## **Model Experiments**

Several model families were trained and compared, including:

### **CNN-Based Models**

* ResNet18 / ResNet34 / ResNet50
* DenseNet121 / DenseNet169
* EfficientNet series

### **Transformer-Based Models**

* ViT (Vision Transformer)
* Swin Transformer

### **Best Performing Model**

The **Vision Transformer (ViT)** achieved the highest performance due to its strength in:

* Capturing long-range dependencies
* Handling AU-based feature embeddings
* Smooth fine-tuning on clinical datasets

ViT produced the most stable results and generalized well across patient categories after fine-tuning.



## **Scoring System**

A custom scoring method was developed:

* Each Action Unit contributes a weighted score
* Intensities are normalized across frames
* Thresholds classify the depression level into:

  * Normal
  * Mild
  * Moderate
  * Severe

This makes the system interpretable and clinically meaningful.



## **Technologies Used**

* Python
* PyFEAT (Action Unit Extraction)
* TensorFlow / PyTorch (depending on your model)
* Vision Transformer (ViT)
* ResNet / DenseNet / EfficientNet / Swin Transformer
* NumPy
* OpenCV
* Scikit-learn

