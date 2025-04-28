# Ensemble-Based Intrusion Detection System Using Machine Learning Techniques


## 📄 Project Overview
This project develops a robust Intrusion Detection System (IDS) using machine learning ensemble models to detect and prevent cyber threats like DDoS, port scanning, brute force attacks, and zero-day exploits. By leveraging ensemble techniques such as Bagging, Boosting, and Stacking, the system significantly improves detection accuracy, reduces false positives, and handles class imbalance effectively.

## 🎯 Objectives
- Analyze and compare machine learning models for intrusion detection.
- Address high false-positive rates and generalization challenges.
- Implement ensemble techniques for enhanced threat detection.
- Evaluate performance using the MSCAD dataset.

## ⚙️ Technologies Used
- **Python**
- **Scikit-learn**, **XGBoost**, **Imbalanced-learn (SMOTE)**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Machine Learning Models:** Logistic Regression, SVC, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- **Ensemble Techniques:** Bagging, Stacking

## 📚 Dataset
- **MSCAD (Multi-Step Cyber Attack Dataset)**
  - 128,799 samples
  - Attack types: Brute Force, Port Scan, HTTP DDoS, ICMP Flood, Web Crawling, and Normal traffic.

## 🛠️ Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/ensemble-ids-detection.git](https://github.com/ChrisRogen/DDoS-Intrusion-Detection-Ensemble-Algorithms
    cd DDoS-Intrusion-Detection-Ensemble-Algorithms
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main Python script:
    ```bash
    python main.ipynb
    ```

## 📈 Model Performance Summary

| Model                  | Accuracy | Precision | Recall | F1-Score | FPR   |
|-------------------------|----------|-----------|--------|----------|-------|
| Logistic Regression     | 98.82%   | 98.81%    | 98.81% | 98.81%   | 0.23% |
| Support Vector Classifier (SVC) | 99.33% | 99.31% | 99.31% | 99.31% | 0.13% |
| Decision Tree Classifier | 99.81% | 99.81% | 99.81% | 99.81% | 0.04% |
| Random Forest Classifier | 99.81% | 99.81% | 99.81% | 99.81% | 0.04% |
| Gradient Boosting Classifier | 99.75% | 99.75% | 99.75% | 99.75% | 0.05% |
| XGBoost Classifier | 99.76% | 99.75% | 99.75% | 99.75% | 0.05% |
| Bagging Ensemble | 99.86% | 99.84% | 99.86% | 99.85% | 0.028% |
| Stacking Ensemble | **99.88%** | **99.86%** | **99.88%** | **99.87%** | **0.024%** |

## 📊 Visual Results
- Performance metric comparisons (Accuracy, Precision, Recall, F1-Score)
- Confusion matrices for model evaluations
- ROC-AUC curves for classifier evaluation

## 🔥 Key Highlights
- Up to **99.88% accuracy** with ensemble models.
- 75% reduction in analyst investigation time.
- 60% faster mean time to detect advanced threats.
- 40% lower false positive rate compared to traditional IDS.

## 🚀 Future Work
- Integration of Explainable AI (XAI) for model transparency.
- Real-time deployment on edge devices using lightweight ensembles.
- Adversarial robust ensemble training.
- Federated learning for privacy-preserving IDS.

## 👩‍💻 Author
**I R Chris Rogen**  
B.Tech Computer Engineering  
Karunya Institute of Technology and Sciences

---



