**NAME:** AISHWARYA RAJ
**COMPANY:** CODETECH IT SOLUTIONS
**ID:** CT08ESQ
**DOMAIN** MACHINE LEARNING
**DURATION:** 20TH DECEMBER 2024 - 20TH JANUARY 2025
**MENTOR:** NEELA SANTHOSH
# LINEAR-REGRESSION-ON-HOUSING-PRICES

# Sentiment Analysis on IMDB Movie Reviews

## Project Overview
This project applies Natural Language Processing (NLP) techniques to perform sentiment analysis on movie reviews from the IMDB dataset. The goal is to classify reviews as either positive or negative using machine learning, specifically the Naive Bayes classifier.

## Objective
- Build a text classification model to predict the sentiment of movie reviews.
- Use TF-IDF vectorization to convert textual data into numerical format.
- Visualize common words in positive and negative reviews using word clouds.
- Evaluate model performance using accuracy, confusion matrix, and classification report.

## Dataset
- **Source**: IMDB Movie Reviews Dataset
- **Format**: CSV file containing two columns:
  - `review` – The text of the movie review.
  - `sentiment` – Binary label ('positive' or 'negative') for each review.

## Technologies and Libraries
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy – Data manipulation and analysis
  - Matplotlib, Seaborn – Visualization
  - Scikit-learn – Machine learning
  - NLTK – Natural language processing
  - WordCloud – Word cloud generation

## Implementation
### Key Steps:
1. **Data Loading and Exploration**
   - The IMDB dataset is loaded into a Pandas DataFrame.
   - Initial data exploration includes displaying summary statistics and the first few rows.

2. **Text Preprocessing**
   - Text is cleaned by:
     - Lowercasing the text
     - Removing punctuation and non-alphabetic characters
     - Filtering stopwords (using NLTK's stopword list)
   - Each review is processed using the `clean_text` function, which applies these transformations.

3. **Word Cloud Visualization**
   - Word clouds are generated to visualize the most frequent words in positive and negative reviews.

4. **Feature Extraction (TF-IDF)**
   - Reviews are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) with a maximum of 5000 features.

5. **Model Training and Evaluation**
   - The dataset is split into training and test sets (80-20 split).
   - A Naive Bayes classifier (`MultinomialNB`) is trained on the training data.
   - The model is evaluated using accuracy, confusion matrix, and a classification report.

## Results
- **Accuracy** – The model achieves a competitive accuracy in classifying movie reviews.
- **Classification Report** – Shows precision, recall, and F1-score for both positive and negative reviews.
- **Confusion Matrix** – Visualizes the model's performance in predicting sentiments.

## Project Structure
```
|-- sentiment_analysis_imdb
    |-- data
        |-- IMDB Dataset.csv
    |-- notebooks
        |-- data_exploration.ipynb
    |-- src
        |-- sentiment_analysis.py
    |-- results
        |-- positive_wordcloud.png
        |-- negative_wordcloud.png
        |-- confusion_matrix.png
    |-- README.md
```

## How to Run the Project
1. Clone the repository:
```
git clone https://github.com/username/sentiment_analysis_imdb.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Python script:
```
python sentiment_analysis.py
```

## Future Enhancements
- Experiment with different classifiers (Logistic Regression, SVM, etc.).
- Implement LSTM or Transformer models for improved accuracy.
- Deploy the model as a web app using Flask or FastAPI.

## Conclusion
This project successfully demonstrates how to apply machine learning to perform sentiment analysis on textual data. By combining NLP techniques and visualization, the project provides valuable insights into the sentiment behind movie reviews.

