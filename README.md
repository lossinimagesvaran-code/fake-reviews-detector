Fake-Reviews-Detection

A machine learning project to detect fake (computer generated) reviews vs genuine (human written) reviews using multiple classification algorithms.

Project Overview

This project aims to classify reviews as either:
- CG: Computer Generated (Fake Reviews)
- OR: Original Reviews (Human-generated, authentic)

The system uses natural language processing techniques and multiple machine learning algorithms to achieve high accuracy in detecting fraudulent reviews.

Dataset

The dataset contains 40,432 reviews with the following features:
- category: Product category (Home_and_Kitchen_5, etc.)
- rating: Numerical rating (1.0 to 5.0)
- label: CG (Computer Generated) or OR (Original Review)
- text_: The review text content

Machine Learning Models Used

The project implements and compares 6 different classification algorithms:
1. Logistic Regression - 86% accuracy
2. K-Nearest Neighbors (KNN) - 58% accuracy
3. Decision Tree Classifier - 73% accuracy
4. Random Forest Classifier - 84% accuracy
5. Support Vector Classifier - 88% accuracy (Best Performer)
6. Multinomial Naive Bayes - 84% accuracy

Features

- Text Preprocessing: Punctuation removal, stopword elimination, stemming, lemmatization
- Feature Extraction: CountVectorizer (Bag of Words) and TF-IDF Transformer
- Multiple Algorithms: Compare performance across 6 classifiers
- Model Persistence: Save and load trained models
- Batch Prediction: Analyze multiple reviews at once

