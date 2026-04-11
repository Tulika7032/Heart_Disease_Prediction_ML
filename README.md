# 🫀 CardioScan — Heart Disease Prediction System

An **end-to-end Machine Learning project** that predicts the **likelihood** of **heart disease** using **patient clinical data**, featuring a **modular training pipeline**, rigorous **model evaluation**, and an **interactive Streamlit application** for **real-time risk assessment.**

The system enables early **risk identification** by providing **probability-based predictions**, helping demonstrate how ML can support **data-driven decision-making** in **healthcare scenarios.**

---

## Project Overview

CardioScan demonstrates the complete **ML lifecycle**:
- **Data preprocessing**  
- **Model training & evaluation**  
- **Best model selection**  
- **Deployment via web interface**  

---

##  Key Highlights

- Built a **modular ML pipeline**
- **Trained multiple models:**
  - Random Forest
  - Logistic Regression
- **Selected best model using F1-score**
- **Implemented full evaluation:**
  - Accuracy, Precision, Recall, F1, ROC-AUC
- Developed an interactive **Streamlit dashboard**
  
---

## Tech Stack

**Languages & Libraries**
- **Python**
- **NumPy, Pandas**
- **Scikit-learn**
- **Joblib**

**Frontend / App**
- **Streamlit**

---

## Machine Learning Workflow

### 1. Data Preprocessing
- **Load dataset**
- **Split into features and target**
- **Train-test split (80/20)**

### 2. Model Training
**- Pipeline:**
  - **StandardScaler + RandomForest**
  - **StandardScaler + LogisticRegression**

### 3. Model Evaluation
**- Metrics:**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC**

### 4. Model Performance

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|----------|--------|----------|--------|
| Random Forest          | 83.6%    | 0.84     | 0.84   | 0.84     | 0.92   |
| Logistic Regression    | 85.2%    | 0.87     | 0.84   | 0.86     | 0.93   |

### 5. Model Selection
- Best model selected using **F1-score**
- **Saved as `model.pkl`**

### 6. Prediction
- **Load saved model**
- **Predict on new input data**

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-project.git
cd heart-disease-project
```

### 2. Create a Virtual Enviornment
```bash
python -m venv venv
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python main.py
```

### 5. Run the streamlit app
```bash
streamlit run app.py
```
--- 

## Application Features
- **Interactive patient input panel**  
- **Real-time health metrics visualization**  
- **ML-based heart disease risk prediction**  
- **Probability-based risk scoring**  
- **Clean and intuitive Streamlit UI**  

