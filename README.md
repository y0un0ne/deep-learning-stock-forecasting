# 📈 Stock Price Prediction — BPNN, CNN, and LSTM

A comparative deep learning study on stock price forecasting using three architectures: **Backpropagation Neural Network (BPNN)**, **Convolutional Neural Network (CNN)**, and **Long Short-Term Memory (LSTM)**. The BPNN is implemented **from scratch using NumPy only** (no deep learning frameworks), while the CNN and LSTM use TensorFlow/Keras.

---

## 🗂️ Project Structure

```
├── BPNNPred-tanh_Stock_Market_Prediction.ipynb   ← Best model (tanh · NumPy)
├── ReLu_Stock_Market_Prediction.ipynb            ← BPNN variant with ReLU
├── Sigmoid_Stock_Market_Prediction.ipynb         ← BPNN variant with Sigmoid
├── CNNPred.ipynb                                 ← CNN model (Keras)
├── LSTMPred.ipynb                                ← Bidirectional LSTM (Keras)
├── best_cnn_model.keras                          ← Saved CNN weights
├── best_lstm_model.keras                         ← Saved LSTM weights
└── dataset.csv                                   ← Historical stock price data
```
## Technologies Used
### Core Language
- Python 3

### Data & Preprocessing
- Pandas — loading and sorting the CSV dataset, datetime conversion
- NumPy — all matrix operations, especially the from-scratch BPNN math
- scikit-learn — MinMaxScaler for normalisation, train_test_split for data splitting, and mean_absolute_error / mean_squared_error for evaluation metrics

### Deep Learning
- TensorFlow / Keras — building, training, and saving the CNN and LSTM models (Conv1D, LSTM, Bidirectional, Dense, BatchNormalization, Dropout, callbacks like EarlyStopping and ModelCheckpoint)

### Visualisation
- Matplotlib — all training loss curves and predicted vs actual price plots

---

## 🧠 Models & Approach

All models solve the same task: **given the last 50 days of Adjusted Close price, predict the next day's price** (sliding window regression).

### BPNN (Built from Scratch)
- **Architecture:** Input → Dense(128) → Dense(64) → Dense(32) → Output(1)
- **Activation (hidden):** Tanh (final), with ReLU and Sigmoid variants for comparison
- **Activation (output):** Linear
- **Optimiser:** SGD (sample-by-sample, no mini-batch)
- **Weight init:** Xavier for Tanh/Sigmoid · He for ReLU
- **Input:** Flattened 50-day window → (50,) vector

### CNN (TensorFlow/Keras)
- **Architecture:** Conv1D(64, k=5) → BN → Pool → Dropout → Conv1D(32, k=3) → BN → Pool → Dropout → Dense(32) → Output(1)
- **Activation:** ReLU (hidden), Linear (output)
- **Optimiser:** Adam · Callbacks: EarlyStopping, ReduceLROnPlateau
- **Input:** (50, 1) 3-D sequence

### LSTM (TensorFlow/Keras)
- **Architecture:** BiLSTM(64) → BiLSTM(32) → Dense(64, ReLU) → Output(1)
- **Optimiser:** Adam · Callbacks: EarlyStopping, ReduceLROnPlateau
- **Input:** (50, 1) 3-D sequence

---

## 📊 Results Summary

| Model | Activation | MSE (denorm.) | MAE (denorm.) |
|-------|-----------|:-------------:|:-------------:|
| **BPNN** | **Tanh** | **~50.28** | **~37.84** |
| CNN | ReLU | ~171.52 | ~110.06 |
| LSTM | Tanh + ReLU | ~183.80 | ~125.60 |

> **Key finding:** The simpler BPNN outperformed both CNN and LSTM. With a single univariate input feature and a relatively small dataset, the more complex architectures were prone to underfitting/overfitting, while the feedforward BPNN captured the temporal patterns effectively via the sliding window approach.

---

## 🗃️ Dataset

The dataset (`dataset.csv`) contains daily historical stock prices with the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.

- The BPNN (tanh) model uses only the `Adj Close` column.
- The BPNN (ReLU & Sigmoid) variants use all 6 numerical features.
- All models normalise features to `[0, 1]` using `MinMaxScaler` before training.

Data split: **70% train / 15% validation / 15% test** (chronological, no shuffling to avoid data leakage).

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install numpy pandas matplotlib scikit-learn tensorflow
```

Then open any notebook in Jupyter:

```bash
jupyter notebook
```

> The BPNN notebooks require **only NumPy, Pandas, and Matplotlib** — no deep learning framework needed.

---

## 🚀 Quickstart

To reproduce the best result (BPNN with Tanh):

1. Place `dataset.csv` in the same directory as the notebooks.
2. Open `BPNNPred-tanh_Stock_Market_Prediction.ipynb`.
3. Run all cells top to bottom.

To load and run inference with the saved CNN or LSTM model:

```python
from tensorflow.keras.models import load_model

cnn_model  = load_model('best_cnn_model.keras')
lstm_model = load_model('best_lstm_model.keras')
```

---

## 📚 Background

This project was developed as a final assignment for one of the courses i took. It explores the practical trade-offs between model complexity and performance on financial time-series data.

Key topics covered:
- Time-series forecasting with sliding windows
- Forward & backward propagation (manual NumPy implementation)
- Convolutional feature extraction for temporal data
- Bidirectional LSTM for sequence modelling
- Hyperparameter comparison: activation functions, optimisers, learning rate

---

## 📄 References

1. Baihaqi et al., "Unveiling the Precision of Deep Learning Models for Stock Price Prediction," ICCTEIE 2023.
2. Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors," *Nature*, 1986.
3. Hochreiter & Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
4. Karim et al., "LSTM Fully Convolutional Networks for Time Series Classification," *IEEE Access*, 2018.
