import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    pass


def clean_data(df):
    messages = pd.read_csv('data/disaster_messages.csv')
    categories = pd.read_csv('data/disaster_categories.csv')
    df = messages.merge(categories, how='inner')
    categories = df['categories'].str.split(pat=';', expand=True)

    row = categories.head(1)
    
    extract_col_names = lambda x: x.str.split('-')[0][0]
    category_colnames = categories.apply(extract_col_names)
    
    categories.columns = category_colnames

    extract_bool = lambda x: int(x.split('-')[1])
    for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(extract_bool)

    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1) # drop the original categories column from `df`
    df = pd.concat([categories, df], axis=1) # concatenate the original dataframe with the new `categories` dataframe

    # Remove duplicates
    duplicates_bool = df.duplicated()
    df = df[~duplicates_bool] # drop duplicates
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///data/disaster_db.db')
    df.to_sql('data', engine, index=False)
    pass


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()