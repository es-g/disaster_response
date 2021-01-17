import gzip
import pickle
import pickletools
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from xgboost import XGBClassifier

nltk.download(['stopwords', 'wordnet', 'punkt'])


def load_data(database_filepath):
    """
    Reads data from SQL database into pandas DataFrame, splits by X, y and category_names

    :param database_filepath: filepath to SQL database
    :type database_filepath: str

    :return:
            X: feature variable (message)
            y: target variables' values (0 or 1)
            category_names: names of categories

    :rtype X: array_like
    :rtype y: int
    :rtype category_names: str
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('data', engine)  # Load data from database to pandas DataFrame
    categories = df.select_dtypes(include=['int64'])  # Select only int64 datatypes
    categories = categories.drop('id', axis=1)  # Drop id column as irrelevant
    X = df['message'].values
    y = categories.values
    category_names = list(categories.columns)

    return X, y, category_names


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
    cache_dir = "."
    xgb = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1,
                                                        scale_pos_weight=5),
                                n_jobs=-1)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', xgb)
    ], memory=cache_dir, verbose=True)

    return pipeline


def build_model_gridSearch():
    cache_dir = "."
    xgb = MultiOutputClassifier(estimator=XGBClassifier(n_jobs=-1,
                                                        scale_pos_weight=5),
                                n_jobs=-1)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', xgb)
    ], memory=cache_dir, verbose=True)

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)  # Predict model

    for i in range(len(category_names)):
        print("Category: {}\n Accuracy: {:.3f}\t Precision: {:.3f}\t Recall: {:.3f} \n\n".format(
            category_names[i],
            accuracy_score(Y_test[:, i], y_pred[:, i]),
            precision_score(Y_test[:, i], y_pred[:, i]),
            recall_score(Y_test[:, i], y_pred[:, i])
            )
        )


def save_model(model, model_filepath):

    with gzip.open(model_filepath, "wb") as f:
        pickled = pickle.dumps(model)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath = sys.argv[1:3]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        if sys.argv[3] == '1':
            # Grid Search
            print('Building model with GridSearch...')
            model_cv = build_model_gridSearch()
            print(model_cv.get_params())

            print('Training model GridSearch...')
            model_cv.fit(X_train, Y_train)

            print("\nBest Parameters:", model_cv.best_params_)

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to ' 
              'save the model to as the second argument and 0 or 1 as the third argument. \n'
              '(0 to skip Grid Search, 1 to run Grid Search)'
              ' \nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl 0')


if __name__ == '__main__':
    main()
