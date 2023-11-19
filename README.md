# ```                AI & ML Project                ```

# DeepFake Detection using Convolutional Neural Networks
This repository contains the source code and documentation for a DeepFake detection project. The project leverages machine learning techniques, specifically a convolutional neural network (CNN) based on the MobileNetV2 architecture, to identify and distinguish between authentic and manipulated images.

## Key Features

- **Data Loading and Preprocessing:** Real and DeepFake images are loaded and preprocessed using OpenCV, ensuring a standardized format for analysis.
- **Data Augmentation:** ImageDataGenerator is employed for on-the-fly data augmentation during training to enhance model generalization.
- **Model Architecture:** MobileNetV2 is used as the base model for feature extraction. The custom neural network includes global average pooling, dense layers, dropout for regularization, and a sigmoid layer for binary classification.
- **Training and Evaluation:** The model is trained using TensorFlow, and rigorous evaluations are conducted on a separate testing dataset to assess performance.
- **Model Persistence:** The trained model is saved for future use, facilitating deployment for real-time DeepFake detection.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Other dependencies (install using `pip install -r requirements.txt`)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/NoorMahammad-S/deepfake-detection.git
    cd deepfake-detection
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:

    ```bash
    python your_main_script.py
    ```

## Usage

1. Customize the paths to your real and DeepFake image datasets in `your_main_script.py`.
2. Run the script to train the model and save it for future use.
3. Experiment with different hyperparameters and model architectures for potential improvements.

## Contact

- Noor Mahammad
- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/noor-mahammad/)
