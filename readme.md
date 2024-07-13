# Mushroom Classification Project

## Overview

This project uses machine learning to classify mushrooms as either edible or poisonous based on their physical characteristics. The dataset used for this project is the Mushroom dataset, which contains various features of mushrooms, including their cap shape, cap surface, gill color, and more. The goal is to build a neural network that can accurately predict whether a mushroom is edible or poisonous.

## Requirements

To run this project, you need the following dependencies:
- Python 3.6+
- PyTorch
- pandas
- scikit-learn

You can install the required packages using:
```bash
pip install torch pandas scikit-learn
```

## Dataset

The dataset used is `mushrooms.csv`, which should be placed in the root directory of the project. Each row in the dataset represents a single mushroom, with various features and a class label indicating whether it is edible or poisonous.

## Preprocessing

The categorical features in the dataset are encoded using `LabelEncoder` from scikit-learn. The features (`X`) and labels (`y`) are then converted to PyTorch tensors.

## Model Architecture

The model used is a simple feedforward neural network with the following architecture:
- Input layer: Size equal to the number of features
- Hidden layer: 16 neurons with ReLU activation
- Output layer: 2 neurons (edible or poisonous) with softmax activation

## Training

The model is trained using the following settings:
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Number of epochs: 50
- Batch size: 32

During training, the model's weights are updated based on the loss computed between the predicted and actual labels. The training progress is printed out for each epoch.

## Evaluation

After training, the model is evaluated on the test set to determine its accuracy. The accuracy is printed out as a percentage.