import gzip
import json
import os
import pickle

import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
def find_filename(file_ex):
    """
    Returns first filename with specified extension. E.g. '.db', '.pkl'
    :return: file - filename
    """

    filenames = os.listdir('../data')
    filenames.extend(os.listdir('../models'))
    for file in filenames:
        if file.endswith(file_ex):
            return file


engine = create_engine('sqlite:///../data/{}'.format(find_filename('.db')))

df = pd.read_sql_table('data', engine)
categories = df.select_dtypes(include=['int64'])  # Select only int64 datatypes
categories = categories.drop('id', axis=1)  # Drop id column as irrelevant

# load model
with gzip.open("../models/{}".format(find_filename('.pkl')), 'rb') as f:
    p = pickle.Unpickler(f)
    model = p.load()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    mean_categories = categories.mean()
    category_names = categories.columns
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=mean_categories
                )
            ],

            'layout': {
                'title': 'Distribution of messages categories',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': ""
                }
            }
        },

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(categories, classification_labels))
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3001, debug=True)
