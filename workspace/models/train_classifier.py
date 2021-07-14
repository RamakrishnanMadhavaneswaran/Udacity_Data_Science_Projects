import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import pickle


def load_data(database_filepath):
    '''
    Function to load data from sql lite database
    
    ARGS:
    database_filepath: path to sqlite database file
    
    OUTPUT:
    X: message column
    y: 36 output categories
    column_names_of_y: names of output categories
    
    '''
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response_data', engine)
    
    X = df.message.values
    y = df.iloc[:,4:] # All columns except the first column
    column_names_of_y = list(y.columns)
    return X,y,column_names_of_y


def tokenize(text):
    '''
    This function helps to tokenize words
    
    ARGS:
    text: message to be word tokenized
    
    OUTPUT:
    clean_tokens: cleaned word tokens of the message
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Function to build the model (pipeline with estimator).
    
    ARGS:
    No Arguments
    
    OUTPUT:
    cv: output model
    
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('multi',MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'multi__estimator__criterion': ['gini'],
        'multi__estimator__max_depth': [None],
        'multi__estimator__n_estimators': [1]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to print the evaluated results of the trained model on test data
    
    ARGS:
    model: trained model
    X_test: test portion of the data
    Y_test: test portion of the output data
    category_names: output category names
    
    OUTPUT:
    No Output
    
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    
    print (Y_test.shape)
    print (y_pred.shape)
    print(classification_report(Y_test,y_pred,target_names=category_names))


def save_model(model, model_filepath):
    '''
    Function to save the model as pickle file
    
    ARGS:
    model: trained model
    model_filepath: path and filename to save the pickle file
    
    OUTPUT:
    No output
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
