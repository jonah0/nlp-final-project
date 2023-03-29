# Topic Analysis

The goal of this project was to build a classifier that could identify the topic of a piece of text as being either book, movie, or resaturant. We used a linear support vector machine (SVM) model with a TfidfVectorizer to convert text into feature vectors.

## Method

We collected three separate datasets, one for each of the desired topics, and compiled them into a single Pandas DataFrame. We then fitted and transformed the training text using the TfidfVectorizer provided by the sklearn library. We trained a linear SVM model on the vectorized training data and used it to predict the labels of the test dataset, which we also transformed using the same TfidfVectorizer. (https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)

## Results

Our model performed well, achieving an accuracy of 90%. The classification report shows that the model had a perfect precision and recall for book, and a high precision and recall for movie, but struggled somewhat with the restaurant category.

We then compared the predicted labels with the actual topic labels for each text. We found that the model correctly predicted the topic for the majority of the texts, with only one misclassification in each category.

## Classification Report

```
              precision    recall  f1-score   support

        book       1.00      1.00      1.00         2
       movie       0.83      1.00      0.91         5
  restaurant       1.00      0.67      0.80         3

    accuracy                           0.90        10
   macro avg       0.94      0.89      0.90        10
weighted avg       0.92      0.90      0.89        10
```

## Analysis

The method used for topic analysis in this study is supervised machine learning. Specifically, a linear SVM model was trained on text data that was first transformed using the TfidfVectorizer. This method is commonly used in text classification tasks such as topic analysis. The TfidfVectorizer converts text data into a numerical representation, making it easier for machine learning algorithms to process and classify the data. The SVM model is a type of machine learning algorithm that is often effective for text classification tasks.

Our model's performance was likely due to the distinct language patterns and vocabulary used in each topic. For example, movie reviews may frequently include mentions of the actors and director, while restaurant reviews may frequently mention the type of cuisine and quality of service. Our book reviews dataset seems to include the word "book" in the vast majority of training instances, which is likely lending to a bias. The model may not classify text as book unless it explicitly says it is about a book. In contrast, the movie review dataset was much larger and more diverse, so the model tends to classify more diverse texts as movie. This is likely what caused the one misclassification in the restaurant category -- the restaurant review did not explicitly mention the word "restaurant" or "food", instead using the less-common word "diner". Therefore, the model classified it as movie instead.

There were some other limitations to our approach. We used a simple linear SVM model and did not perform any hyperparameter tuning or optimization. Further experimentation could include the use of more complex models or the incorporation of additional features, such as sentiment analysis or part-of-speech tagging.
