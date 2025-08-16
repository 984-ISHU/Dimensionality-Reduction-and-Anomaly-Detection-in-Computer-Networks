# Dimensionality Reduction and Anomaly Detection in Network Traffic Data Using Autoencoders

## 📌 Project Overview

In today's digital landscape, monitoring network traffic is crucial to ensure the security and efficiency of data flow. With the growing complexity of network systems, traditional anomaly detection methods often struggle with high-dimensional data. This project demonstrates how **autoencoders**, a type of unsupervised neural network, can be applied for:

* **Dimensionality Reduction** → compressing network traffic data into a lower-dimensional representation.
* **Anomaly Detection** → identifying unusual patterns or potential threats using reconstruction error.

The approach improves scalability and robustness in detecting anomalies, such as network intrusions or irregular behaviors.

---

## 🎯 Objectives

1. **Dimensionality Reduction**: Simplify high-dimensional traffic data while retaining essential features.
2. **Anomaly Detection**: Detect anomalies using reconstruction errors from the autoencoder.
3. **Evaluation & Visualization**: Assess model performance using F1-score, ROC AUC, confusion matrix, and visualizations.

---

## 📂 Dataset

* **Source**: [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
* **Files Used**:

  * `UNSW_NB15_training-set.csv`
  * `UNSW_NB15_testing-set.csv`
* Data is merged and pre-processed before applying ML models.

---

## ⚙️ Workflow

1. **Import Libraries** → Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras.
2. **Load Dataset** → Training and Testing sets are read and merged.
3. **Preprocessing** → Standardization, handling categorical features, and data cleaning.
4. **Autoencoder Model** →

   * Input layer with dense encoding layers
   * Batch normalization & dropout for regularization
   * Reconstruction of input
5. **Training & Validation** → Early stopping & model checkpointing.
6. **Evaluation** →

   * ROC AUC, F1-score, accuracy
   * Confusion matrix
   * Visualization of reconstruction errors

---

## 🛠️ Installation

```bash
# Clone repository
git clone <repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```

Ensure the dataset CSV files are placed in the correct directory as specified in the notebook.

---

## 📊 Results

* The autoencoder effectively reduced dimensionality while retaining essential patterns.
* High reconstruction errors correlated with anomalous network traffic.
* Performance was evaluated using standard classification metrics.

---

## 🚀 Future Work

* Extend to real-time streaming data.
* Compare with other anomaly detection methods (Isolation Forest, PCA, GANs).
* Optimize autoencoder architecture for faster training.

---

## 📜 License

This project is for **educational and research purposes**. Please check dataset license before commercial use.
