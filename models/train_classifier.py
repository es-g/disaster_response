import sys
from typing import Union, Iterator
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
nltk.download(['stopwords', 'wordnet', 'punkt'])


def load_data(database_filepath):
    """
    Reads data from SQL database into pandas DataFrame, splits by X and Y

    :param database_filepath: filepath to SQL database
    :return: X, Y: feature and target variables
    """
    engine = create_engine('sqlite:/// {}'.format(database_filepath))
    df = pd.read_sql('data', engine)  # Load data from database to pandas DataFrame
    categories = df.select_dtypes(include=['int64'])  # Select only int64 datatypes
    categories = categories.drop('id', axis=1)  # Drop id column as irrelevant
    X = df['message']
    Y = categories

    return X, Y


def tokenize(text):
    """
    Tokenizes text by normalizing, lemmatizing, removing stop words and white spaces

    :param text: Raw text
    :return: words: Tokenized list of words
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())  # Normalize text and remove white space
    words = word_tokenize(text)  # Tokenize text
    words = [w for w in words if w not in stopwords.words("english")]  # Remove stop words
    words = [WordNetLemmatizer().lemmatize(w) for w in words]  # Lemmatize
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]  # Lemmatize verbs by specifying pos

    return words


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
