# Credit Card Fraud Detection: A Comparison of Machine Learning and Deep Learning Models

## Abstract

This project aims to develop and compare various machine learning and deep learning models for credit card fraud detection. The dataset used for this project is sourced from Kaggle and contains credit card transactions, including both legitimate and fraudulent transactions. The objective is to build robust models that can effectively identify fraudulent transactions, and to compare the performance of different techniques.

### Dataset

The dataset used for this project can be found on Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It comprises a substantial number of transactions, each described by features like transaction amount, timestamp, and anonymized numerical attributes. The dataset is imbalanced, with only a small fraction of transactions being fraudulent.

### Approach

1. **Data Preprocessing:** Extensive data preprocessing is performed, encompassing handling missing values, scaling numerical features, and encoding categorical variables.

2. **Feature Engineering:** Relevant features are identified, and new features are generated to potentially enhance model performance.

3. **Exploratory Data Analysis (EDA):** In-depth analysis is conducted to understand the distribution of legitimate and fraudulent transactions, uncover patterns, and create visualizations to guide model selection.

4. **Model Selection:**

   - **Machine Learning Models:** Various machine learning algorithms, such as Random Forest, Support Vector Machines, Logistic Regression, XGBoost, Decision Tree and K-Nearest-Neighbour are employed to create effective fraud detection models.

   - **Deep Learning Models:** Different deep learning architectures, including Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), are explored to harness the power of neural networks.

5. **Model Training:**

   - **Machine Learning Models:** Selected machine learning models are trained on the preprocessed dataset, and hyperparameters are fine-tuned to achieve optimal performance.

   - **Deep Learning Models:** Deep learning architectures are implemented using libraries like TensorFlow or PyTorch. Model training involves setting up appropriate layers, optimizers, and loss functions.

6. **Evaluation:**

   - The performance of each model is evaluated using metrics such as precision, recall, F1-score, and ROC-AUC, taking into account the imbalanced nature of the dataset.

7. **Comparison and Analysis:**

   - The results of machine learning and deep learning models are compared in terms of their effectiveness in detecting credit card fraud.

   - Consideration is given to factors like model complexity, computational resources required, and training time.

### Conclusion

This project aims to build, compare, and analyze various machine learning and deep learning models for credit card fraud detection. By utilizing the provided dataset and applying extensive preprocessing, feature engineering, and model selection strategies, we seek to create models that can accurately distinguish between legitimate and fraudulent transactions. The comparison of different techniques will provide insights into the strengths and weaknesses of each approach, contributing to a better understanding of fraud detection in financial systems. Ultimately, the findings of this project can guide financial institutions in selecting appropriate models to enhance security and mitigate credit card fraud risks.
