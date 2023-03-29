import nltk
import pandas as pd
import spacy
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS

# TEXT MINING FINAL PROJECT -- TOPIC CLASSIFICATION

# prepare training dataset
print("Compiling training datasets...")
movie_df = pd.read_csv('./topic-datasets/movie_reviews.csv')
movie_df['topic'] = 'movie'
movie_df

restaurant_df = pd.read_csv(
    './topic-datasets/restaurant-reviews.tsv', sep='\t')
restaurant_df['text'] = restaurant_df['Review']
restaurant_df['topic'] = 'restaurant'
restaurant_df

book_df = pd.read_csv('./topic-datasets/book-reviews.csv')
book_df['text'] = book_df['ReviewContent']
book_df['topic'] = 'book'
book_df

# even out the number of instances of each topic
lowest_num_rows = min([len(movie_df), len(restaurant_df), len(book_df)])
mixed_df = pd.concat([
    movie_df[['text', 'topic']].sample(lowest_num_rows, axis='index'),
    restaurant_df[['text', 'topic']].sample(lowest_num_rows, axis='index'),
    book_df[['text', 'topic']].sample(lowest_num_rows, axis='index'),
])
mixed_df.to_csv('./topic-datasets/mixed-reviews.csv')


print("Reading in topic classification training set...")
dataset = pd.read_csv('./topic-datasets/mixed-reviews.csv')
print(dataset)

print("Fitting and transforming text using TfidfVectorizer...")
vectorizer = TfidfVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
vectorizer.fit(dataset['text'])  # fit vectorizer to all data

# split dataframe into train and test data
train, test = train_test_split(dataset, test_size=0.1)

# vectorize data
train_vectors = vectorizer.transform(train['text'])
test_vectors = vectorizer.transform(test['text'])

# train a linear classifier
print("Training linearSVC on vectorized training data...")
lsvc = svm.LinearSVC()
lsvc.fit(train_vectors, train['topic'])

# make predictions and generate classification report
print("Evaluating model against our test dataset...")
y_pred = lsvc.predict(test_vectors)
print(classification_report(test['topic'], y_pred))

# make predictions and generate classification report for given assignment test data
print("Reading in given test dataset...")
official_test_df = pd.read_csv(
    './test-datasets/sentiment-topic-final-test.tsv', sep='\t')
print(official_test_df)
official_test_vectors = vectorizer.transform(official_test_df['text'])


print("Evaluating model against given assignment dataset...")
official_y_pred = lsvc.predict(official_test_vectors)
print(classification_report(official_test_df['topic'], official_y_pred))


print("Given test dataset: topic (gold) vs topic_pred (predicted)")
official_test_df['topic_pred'] = official_y_pred
print(official_test_df)
