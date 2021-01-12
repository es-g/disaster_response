import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re

def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files and return the merged data as a DataFrame

    :param messages_filepath: filepath to disaster_messages.csv file
    :param categories_filepath: filepath disaster_categories.csv file
    :return: merged DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the messages and categories datasets using the common id
    df = messages.merge(categories, how='inner')

    return df


def clean_data():
    """
    Cleans original DataFrame and returns cleaned DataFrame

    :return: cleaned DataFrame
    """
    df = load_data('disaster_messages.csv', 'disaster_categories.csv')
    # Split the values in the `categories` column on the `;` character
    categories = df['categories'].str.split(pat=';', expand=True)
    # Extract a list of new column names for categories
    category_columns = categories.apply(lambda x: x.str.split('-')[0][0])
    # Replace categories column in df with new category columns
    categories.columns = category_columns

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))

    df = df.drop('categories', axis=1)  # drop the original categories column from `df`
    df = pd.concat([categories, df], axis=1)  # concatenate the original dataframe with the new `categories` dataframe

    # Remove duplicates
    duplicates_bool = df.duplicated()
    df = df[~duplicates_bool]
    # Drop rows with elements other than 0 or 1
    categories = df.select_dtypes(include=['int64'])  # Select only int64 datatypes
    categories = categories.drop('id', axis=1)
    indices = categories[np.isin(categories, [0, 1], invert=True)].index
    df = df.drop(index=indices)

    def is_empty_or_blank(msg):
        """ This function checks if given string is empty
         or contain only white spaces"""
        return re.search("^\s*$", msg)

    is_blank = [is_empty_or_blank(elem) for elem in df['messages']]
    ind = []
    for i, res in enumerate(is_blank):
        if res is not None:
            ind.append(i)
    # Drop elements that contain empty message
    df = df.drop(index=ind)

    return df


def save_data(database_filename):
    """
    Saves cleaned data to a database

    :param database_filename:
    :return: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = clean_data()
    df.to_sql('data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        clean_data()

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepath of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
