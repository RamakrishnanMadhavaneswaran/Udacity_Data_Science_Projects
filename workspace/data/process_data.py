import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #Merge the datasets
    df = pd.merge(messages,categories,on=['id'])
    
    return df, categories


def clean_data(df,categories):
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        #print (categories[column])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
        
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)

    # check number of duplicates
    duplicateRowsDF = df[df.duplicated()]
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    #Remove unwanted data from the related column
    df.related.replace(2,1,inplace=True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('disaster_response_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df,categories)
        
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