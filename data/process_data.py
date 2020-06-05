# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    messages = messages[~messages.id.duplicated()]
    
    categories = pd.read_csv(categories_filepath)
    categories = categories[~categories.id.duplicated()]
    
    df = messages.merge(categories, left_on=["id"], right_on=["id"])
    return df

def clean_data(df):
    categories = df.categories.str.split(pat=";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda str: str[0:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
        
        # assure binary values (either 0 or 1)
        categories.loc[categories[column]!=0, column]=1
        
    df.drop(["categories"], axis=1, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df=df[~df.duplicated()]
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessageData', engine, index=False)


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