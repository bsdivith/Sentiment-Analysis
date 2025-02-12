# Sentiment Analysis: CNN vs RNN vs LSTM

## Overview
This project compares different deep learning architectures—**CNN, RNN, and LSTM**—for **sentiment analysis** on text data. The goal is to determine which model performs best in classifying sentiment (e.g., positive, negative, neutral) based on various evaluation metrics.

## Features
- Implementation of **Convolutional Neural Networks (CNN)**, **Recurrent Neural Networks (RNN)**, and **Long Short-Term Memory Networks (LSTM)** for sentiment classification.
- Use of **word embeddings** for text representation.
- Evaluation on a **real-world dataset** with accuracy, precision, recall, and F1-score.
- **Comparison of training times** and model performances.

## Dataset
The project uses a **preprocessed dataset** containing labeled text reviews. The dataset is split into training, validation, and testing sets for model evaluation.

## Installation & Setup
### Prerequisites
- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/bsdivith/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the training script**:
   ```bash
   python train.py --model cnn
   ```
   Replace `cnn` with `rnn` or `lstm` to train different models.

## Model Performance
- **CNN**: Extracts spatial features but struggles with long-term dependencies.
- **RNN**: Captures sequential patterns but may suffer from vanishing gradients.
- **LSTM**: Handles long-term dependencies better and often outperforms CNN/RNN for sentiment analysis.

## Results
The models are evaluated using **accuracy, precision, recall, and F1-score**. Results are visualized through plots in the `results/` directory.

## Future Enhancements
- Implement **Bidirectional LSTM** for improved text representation.
- Experiment with **transformer-based models** (e.g., BERT) for better performance.
- Optimize hyperparameters using **Bayesian optimization**.

## Author
- **Divith B S** ([GitHub](https://github.com/bsdivith) | [LinkedIn](https://www.linkedin.com/in/divith-b-s))

## License
This project is open-source and licensed under the **MIT License**.

