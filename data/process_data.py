import sys
import pandas as pd
from sqlalchemy import create_engine


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


def clean_data(df):
    df = load_data('disaster_messages.csv', 'disaster_categories.csv')
    # Split the values in the `categories` column on the `;` character
    categories = df['categories'].str.split(pat=';', expand=True)

    category_columns = categories.apply(lambda x: x.str.split('-')[0][0])

    categories.columns = category_columns

    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)  # drop the original categories column from `df`
    df = pd.concat([categories, df], axis=1)  # concatenate the original dataframe with the new `categories` dataframe

    # Remove duplicates
    duplicates_bool = df.duplicated()
    df = df[~duplicates_bool]

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = clean_data(df)
    df.to_sql('data', engine, index=False)


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
        print('Please provide the filepath of the messages and categories ' 
              'datasets as the first and second argument respectively, as ' 
              'well as the filepath of the database to save the cleaned data ' 
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv ' 
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
