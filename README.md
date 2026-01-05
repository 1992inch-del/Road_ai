## Remote sensing road and drivable area semantic segmentation project.


# Overview
This project involves the use of remote sensing data to perform semantic segmentation of road and drivable areas. It leverages the U-Net model, a popular deep learning architecture for image segmentation.

# This repository will contain:
1. Dataset preparation:
   Instructions on how to acquire or generate the dataset, its structure, and preprocessing steps.
2. U-Net baseline model training:
   Code and instructions on how to train the U-Net model using the prepared dataset.
3. Evaluation (IoU / F1):
   Model evaluation metrics using Intersection over Union (IoU) and F1 score.
4. Visualization of results:
   Visualization tools for showing the results of the model predictions, such as masks overlaid on input images.

# Project Structure
1. data/: Contains the dataset used for training, valuing and testing.
2. results/: Stores model checkpoints, training logs, and output images.
3. src/: Contains the Python scripts for training, evaluation, and visualization.
4. README.md: This file.
5. gitignore: Conclude the type of files like .tiff, .jpg and so on.

# Setup Instructions
1. Clone the repository:
    git clone https://github.com/1992inch-del/Road_ai.git
    cd Road_ai

2. Install dependencies:
    pip install -r requirements.txt
    

3. Prepare the dataset:
    Download the dataset from [Dataset Link] and place it in the data/ folder.

# Training the Model

To train the U-Net model, follow these steps:

1. Set up the environment:
    pip install -r requirements.txt

2. Prepare the dataset:
    Download the dataset from [Dataset Link].
    Place the dataset into the data/ folder, ensuring the following structure:
      data/
          images/
          labels/

3. Train the model:
    Run the following command:
    python src/train_unet.py --data_path ./data --epochs 30 --batch_size 2 --learning_rate 0.001 --save_path ./results/-

    data_path: Path to the dataset
    epochs: Number of training epochs
    batch_size: Batch size for training
    learning_rate: Learning rate for the optimizer
    save_path: Path to save the model and results

4. Evaluate the model:
    After training, you can evaluate the model performance using:
    python src/eval_unet_patch.py --model_path ./results/best_model.pth --data_path ./data

    This will give you indicator such as IoU and Dice.

5. Visualize the results:
    To visualize the training results and predictions, run:
    python src/visualize_results.py --results_path ./results/

# Note:
1. The training time and results may vary depending on the hardware you are using(My GPU is Nvidia Geforce RTX4060 and my CPU is AMD i5).
2. Make sure to monitor the training process for any potential issues like overfitting or underfitting.

