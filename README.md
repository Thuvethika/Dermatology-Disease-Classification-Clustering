# Dermatology Disease Classification & Clustering

This project analyzes a dermatology dataset (34 attributes, 6 disease classes) using both **supervised** and **unsupervised machine learning** techniques. The dataset includes **clinical** and **histopathological** features, making classification a challenging but meaningful task.

## 📊 Dataset
- 34 attributes (12 clinical, 22 histopathological)  
- 1 target variable (disease class: 6 types)  
- Diseases include: Psoriasis, Seborrheic Dermatitis, Lichen Planus, Pityriasis Rosea, Chronic Dermatitis, Pityriasis Rubra Pilaris  

## 🔎 Models Implemented
- **Model 1: Gradient Descent (custom implementation)**
  - Predicts disease type using *Age*  
  - Implemented GD algorithm from scratch  

- **Model 2: Random Forest**
  - Uses clinical + histopathological features  
  - Best performing classification model  

- **Model 3: k-Nearest Neighbors (kNN)**
  - Classification on both clinical and histopathological features  

- **Model 4 & 5: Clustering (KMeans & Agglomerative Hierarchical Clustering)**
  - Explored natural groupings of diseases  
  - Evaluated using ARI & NMI  

## 🛠️ Tools & Libraries
- Python, NumPy, Pandas, Scikit-learn  
- Matplotlib, Seaborn  

## 📈 Results
- Random Forest achieved the highest accuracy.  
- Gradient Descent on *Age* alone was weak, showing age isn’t a strong predictor.  
- Clustering showed overlaps → diseases share clinical/histopathological features.  

## 🚀 Key Takeaways
- **Random Forest** works best for this dataset.  
- **Clustering** highlights diagnostic complexity in dermatology.  
- Showcases both **supervised & unsupervised** learning approaches.  

