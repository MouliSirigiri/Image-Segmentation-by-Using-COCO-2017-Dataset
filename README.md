#**Image Segmentation Using COCO-2017 Dataset**

#**Overview**

This repository contains the code, data subset, and analysis for an Applied Data Science project on instance segmentation using a subset of the COCO-2017 dataset. The focus is on four object classes: cake, car, dog, and person. Leveraging Mask R-CNN architecture, we train and evaluate a model to generate precise segmentation masks. The project includes exploratory data analysis (EDA), model training in Google Colab, and performance evaluation using Intersection over Union (IoU) metrics.

#**Key goals:**

Perform EDA to understand class distribution, image sizes, and pixel characteristics.
Train Mask R-CNN for multi-class instance segmentation.
Evaluate model performance and discuss challenges like class imbalance.

Insights highlight strong performance on dominant classes (e.g., 'person' IoU=0.75) but lower on underrepresented ones (e.g., 'cake' IoU=0.60), emphasizing the need for dataset balancing.

#**Key Findings**

EDA Insights: 'Person' class dominates (~50% instances); image sizes vary (avg. 640x480); pixel histograms show need for normalization.
Model Performance: Average IoU=0.70 across classes. Breakdown
Class,Train Instances,IoU (Test)
Person,150,0.75
Car,80,0.68
Dog,50,0.65
Cake,20,0.60

Trends: Higher IoU correlates with class frequency; occlusions/scales challenge 'cake'/'dog'.
Visuals: image1.png (bar chart imbalance), image2.png (histograms), image3.png (sample masks), image4.png (IoU bars).

#**Data Sources**

Primary: Subset of COCO-2017 Dataset (train2017/val2017).
Images: 600 total (300 train, 300 val) filtered for 4 classes.
Annotations: Instance segmentation masks in COCO JSON.

Test Set: 30 held-out images for unbiased evaluation.
Reference: Lin et al. (2014) for dataset details.

#**Methods**

EDA: Pandas/Matplotlib for distributions; OpenCV for image stats.
Preprocessing: Resize (256x256), normalize [0,1]; augmentation (flips, rotations) via Keras.
Model: Mask R-CNN (TensorFlow/Keras implementation) with ResNet-50-FPN backbone.
Loss: Combined RPN, classification, mask (binary cross-entropy).
Training: 5 epochs, Adam optimizer (lr=0.001), batch=16.

Evaluation: Mean IoU per class; qualitative mask overlays.
Mitigations: Dropout (0.5) for overfitting; class weights for imbalance.

#**Limitations and Future Work**

Small subset (600 images) limits generalization; scale to full COCO for robustness.
Class imbalance affects rare classes—future: SMOTE-like oversampling or focal loss.
No real-time inference; extend to edge devices or add post-processing (e.g., CRF).
Compare with DeepLabv3+ for semantic segmentation baselines.
