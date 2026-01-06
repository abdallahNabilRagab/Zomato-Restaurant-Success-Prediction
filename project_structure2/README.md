# 🍽️ Zomato Restaurant Success Prediction  
A Complete End-to-End Machine Learning Pipeline  
Built by **Abdallah Nabil Ragab**

---

## 📊 Project Overview

This project predicts whether a restaurant listed in the **Zomato Bangalore Dataset** will be **successful** based on multiple business, operational, and customer-related features.

A full **MLOps-ready pipeline** was implemented:

- Data ingestion & deep cleaning  
- Business rule validation  
- Feature engineering  
- Train/test preparation  
- Model training (5 ML algorithms)  
- Hyperparameter tuning  
- Model selection  
- Production-grade inference  
- Streamlit app for deployment  

Dataset Source (Kaggle):  
Zomato Bangalore Restaurants Dataset  
(Contains 9,000+ restaurants with 21+ features)

---

## 📂 Project Structure

```
project_structure/
│
├── app/
│   └── streamlit_app.py              # Streamlit prediction app
│
├── data/
│   ├── raw/                          # Original raw data
│   ├── preprocessed/                 # Cleaned & engineered datasets
│   └── processed/                    # Model-ready datasets
│
├── models/
│   ├── saved_models/                 # Optimized ML models (pickle)
│   ├── scalers/                      # Saved transformers
│   └── encoders/                     # Saved encoders
│
└── src/
    ├── data_pipeline.py              # Data cleaning + EDA + feature engineering
    ├── inference.py                  # Production inference pipeline
    ├── model.py                      # Model factory (all ML algorithms)
    ├── train.py                      # Training + model evaluation
    ├── utils.py                      # Utilities & shared helpers
    └── __init__.py
```

---

# 📁 Dataset Description

**Source:** [Zomato Dataset on Kaggle](https://www.kaggle.com/datasets/rajeshrampure/zomato-dataset/data)  

The Zomato dataset contains restaurant information such as:

| Column | Description |
|--------|-------------|
| `name` | Restaurant name |
| `location` | Area / district |
| `listed_in(type)` | Dining / Cafés / Quick Bites / etc. |
| `listed_in(city)` | City group |
| `cuisines` | Type of cuisines served |
| `rest_type` | Restaurant business type |
| `approx_cost(for two people)` | Cost estimate |
| `online_order` | Accepts online orders? |
| `book_table` | Table reservation available? |
| `rate` | Customer rating |
| `votes` | Number of votes |
| `phone`, `menu_item` | Optional business fields |

### 🧪 Target Variable  
A success label was engineered:

- **1** → Rating ≥ 3.75  
- **0** → Otherwise

---

# 🔧 Module-by-Module Documentation

---

# 📌 `src/data_pipeline.py` — *Full Data Processing Pipeline*

A production-ready data preparation pipeline performing:

### ✔ 1. Data Loading & Basic Cleaning
- Remove irrelevant columns  
- Normalize rating values (`4.2/5`, `NEW`, `-`)  
- Convert `votes` to numeric  
- Fix cost values with commas (`1,200` → `1200`)  
- Clean text fields (cuisines, rest_type, location)

### ✔ 2. Business Rule Validation
- Remove rows with invalid cuisine/rest types  
- Validate consistency between `location` and `listed_city`

### ✔ 3. Feature Engineering
- Target creation  
- Cuisine grouping into “regional families”  
- Cost/rating/votes binning  
- Operational classification for restaurant types  
- Binary transformations for online/table/menu features  

### ✔ 4. Data Validation
- Missing value detection  
- Duplicates check  
- Target imbalance reporting  

### ✔ 5. Optional EDA
Automatically generates:
- Univariate plots  
- Correlation heatmaps  
- Target-related behaviour  

### ✔ 6. Saving Output
- Writes processed dataset to `data/preprocessed/`  
- Splits into **Train** / **Test** using timestamp naming  

### ▶ Example
```python
from src.data_pipeline import run_pipeline

run_pipeline(
    raw_path="data/raw/zomato.csv",
    save_path="data/preprocessed/zomato_cleaned.csv",
    training=True,
    run_eda=True,
    split_data=True
)
```

---

# 🤖 `src/model.py` — *Model Factory*

A clean factory to generate ML models with optimized hyperparameters.

Supported algorithms:

| Model | Purpose |
|--------|----------|
| Logistic Regression | Lightweight baseline |
| Linear SVM | Strong linear classifier |
| Decision Tree | Rules & interpretability |
| Random Forest | Ensemble-based robustness |
| XGBoost | Best performer for structured data |

### Example:
```python
from src.model import get_xgb_classifier
model = get_xgb_classifier()
```

---

# 🏋️‍♂️ `src/train.py` — *Automated Training Pipeline*

Handles all training steps:

### ✔ Load latest preprocessed data  
### ✔ Preprocess using:
- RobustScaler  
- BinaryEncoder  
- ColumnTransformer pipeline  

### ✔ Train 5 ML models using `GridSearchCV`  
### ✔ Evaluate performance  
### ✔ Save:
- Processor (scaler + encoder)
- Best model for each algorithm  
- All in timestamped folders

### ▶ Run training:
```bash
python src/train.py
```

---

# 🚀 `src/inference.py` — *Production Inference*

Automatically:

1. Loads latest test file  
2. Loads saved processor  
3. Loads all trained models  
4. Predicts & evaluates  
5. Computes metrics:

- Accuracy  
- Precision/Recall/F1  
- ROC-AUC  
- PR-AUC  
- MCC  
- Confusion Matrix  

### ▶ Run inference:
```bash
python src/inference.py
```

---

# 🖥️ `app/streamlit_app.py` — *Interactive Prediction App*

A powerful UI to test models in real time.

### Features:
✔ Upload or auto-load processor & model  
✔ User input form (city, cost, cuisines, votes, etc.)  
✔ Transforms data exactly like training  
✔ Predicts restaurant success  
✔ Displays probabilities  
✔ Debugging mode for developers  

### ▶ Run Streamlit:
```bash
cd app
streamlit run streamlit_app.py
```

---

# 🧰 Utility Functions (`utils.py`)

- File search helpers  
- Timestamp utilities  
- Safe loading wrappers  
- Pretty model naming  
- Logging helpers  

---

# ⚙️ How to Run the Whole Project

### 1. Place raw dataset:
```
data/raw/zomato.csv
```

### 2. Run preprocessing:
```bash
python src/data_pipeline.py
```

### 3. Train models:
```bash
python src/train.py
```

### 4. Run inference:
```bash
python src/inference.py
```

### 5. Launch Streamlit:
```bash
streamlit run app/streamlit_app.py
```

---

# 👨‍💻 **Developer Information**

**Name:** *Abdallah Nabil Ragab*  
**Role:** Data Scientist • ML Engineer • Software Engineer  
**M.Sc. — Business Information Systems**  
**Email:** abdallah.nabil.ragab94@gmail.com  

🔬 Specializes in:  
- Machine Learning & AI  
- Large-scale Recommendation Systems  
- Data Pipelines & MLOps  
- Python software engineering  
- Streamlit apps  
- NLP & classification systems  

📌 *Feedback, collaboration, and feature requests are always welcome.*

---

# ⭐ Summary

This repository provides a completely automated machine learning system:

- Raw → Clean → Features → Train → Evaluate → Inference → Streamlit UI  
- Supports 5 ML models  
- Uses timestamped versioning  
- Production-ready and fully modular  
- Clear pipeline structure for reproducibility  

A powerful, scalable solution for real-world restaurant success prediction 🚀

