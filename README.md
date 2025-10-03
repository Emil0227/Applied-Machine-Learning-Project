# Applied Machine Learning Project – Food Reviews Classification
This project was completed as part of the **Applied Machine Learning module** during my MSc Computer Science studies at the University of London.  
It demonstrates my ability to design and implement a complete **machine learning pipeline**: from data acquisition and preprocessing, to feature selection, model training, evaluation, and interpretation.

---

## 🎯 Project Objective
To build and evaluate classification models on a **real-world dataset** of food product reviews, with the goal of predicting user satisfaction and identifying patterns in consumer feedback.

---

## 📊 Dataset
- Source: [Amazon Fine Food Reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)  
- Size: 568,000+ observations  
- Target: Review recommendation (positive/negative)  
- Key preprocessing steps:
  - Handled missing values and duplicate reviews
  - Normalized and cleaned text data
  - Converted raw text into vectorized features (TF-IDF)

---

## 🔑 Key Steps & Methods
### 1. Feature Engineering
- Created a **term-document incidence matrix**
- Applied **TF-IDF vectorization** to capture semantic weight of words
- Explored **feature importance** to filter high-value terms

### 2. Model Building
Implemented and compared multiple algorithms:
- **Logistic Regression**  
- **Random Forest**  
- **XGBoost**

Optimizations:
- Hyperparameter tuning (grid search, cross-validation)  
- Train/test split with stratification  
- Evaluation with AUC, Precision, Recall, F1-score  

### 3. Evaluation & Insights
- **Random Forest** achieved highest AUC (0.96) and robust generalization across folds  
- Identified trade-offs between interpretability (Logistic Regression) and performance (XGBoost/Random Forest)  
- Explored error cases where models struggled with sarcastic or ambiguous reviews  

---

## 🛠️ Tools & Skills
- **Python (pandas, scikit-learn, XGBoost, matplotlib)**  
- **Natural Language Processing (TF-IDF, word frequency analysis)**  
- **Data Visualization (word clouds, feature distribution plots)**  
- **Model Evaluation (cross-validation, confusion matrices)**  

---

## 🚀 Results
- Built a scalable classification pipeline applicable to real-world business intelligence tasks (e.g., product review mining, sentiment analysis).  
- Achieved **96% AUC** with Random Forest on test data.  
- Produced visualizations highlighting **word importance and sentiment correlation**.  

---

## 📂 Project Structure
