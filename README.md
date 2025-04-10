April 10th, 2025

This document serves as a technical log to detail the progression and methodologies applied throughout the pipeline corrosion detection project.

Over the past few days, I have focused on building a dataset of corrosion instances. To achieve this, I manually annotated 110 images—captured using the BIKE robot in a corroded pipeline environment—using CVAT (Computer Vision Annotation Tool). CVAT is an open-source, web-based platform developed by Intel, widely used in computer vision tasks for annotating images and videos. It supports various formats and facilitates precise object labeling.

A total of over 700 corrosion instances were annotated and saved in COCO format, which is natively supported by CVAT.

To enhance the dataset's diversity and robustness, I implemented data augmentation techniques. After exploring the Albumentations library and its documentation, I selected transformations that would increase dataset variety without compromising critical visual characteristics of corrosion—particularly its color, which plays a key role in identification. Therefore, color-altering transformations were avoided.

The augmentation process was handled by a custom script, augment_coco.py, which applied operations such as rotation, blurring, noise addition, and flipping. This process expanded the dataset to 660 images, encompassing over 4,200 annotated corrosion objects.

Subsequently, I split the dataset into training, validation, and test subsets, using the split_coco.py script:

    75% for training

    10% for validation

    15% for testing

To prepare the data for training with the YOLOv8 segmentation model, I converted the annotations from COCO to YOLO format using coco2yolo.py. This script also automatically generated the data.yaml configuration file, which specifies dataset paths and class names for the YOLO training pipeline.

With the annotations converted and the dataset properly organized, the project is now ready to proceed with training a YOLO-based model for the segmentation and detection of pipeline corrosion.
