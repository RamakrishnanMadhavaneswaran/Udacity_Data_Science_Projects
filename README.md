# Disaster Response Pipeline Project


## Summary:
This project is for categorizing the messages sent during disaster events so that the messages can be forwarded to an appropriate disaster relief agency.

The categorizing of messages are done using a machine learning pipeline. The categorizing of the messages can be seen on a web app. If someone wants to know the message category, they can type of the search field of the web app and then click on "Classify Message" button to see the classified category.


## Installations:

The below libraries are used in this project:

1.) Data Science Libraries
import pandas as pd
import numpy as np

2.) To process Command Line inputs
import sys

3.) To connect to SQL Lite Database
from sqlalchemy import create_engine

4.) Natural Language Processing Libraries
import nltk
nltk.download("punkt")
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

5.) Scikit Learn Libraries for Machine Learning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

6.) To save the machine learning model
import pickle

7.) Libraries to work with JSON data
import json

8.) Python Graphing Library
import plotly
from plotly.graph_objs import Bar

9.) Web application framework Library
from flask import Flask
from flask import render_template, request, jsonify


## File Descriptions:

There are five directories inside the workspace directory:

###1. data
		DisasterResponse.db: SQLite database containing the output of the ETL pipeline
		disaster_categories.csv: categories dataset is available in this file
		disaster_messages.csv: messages dataset is available in this file
		process_data.py: ETL python script that takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database

###2. models
		train_classifier.py: machine learning pipeline script that takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file

###3. app
		run.py: python script that runs the web application
		templates directory: contains html files related to the landing page and, to display the classification results of the model
		
###4. project_snapshots/snapshot_1
		this directory contains the snapshots of the web application
		
###5. Notebooks Used for this project
		ETL Pipeline Preparation.ipynb: Jupyter notebook that takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database
		ML Pipeline Preparation.ipynb: Jupyter notebook that takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file


## Instructions to run the project:
To execute the code, please follow the below steps in Udacity's Workspace
1. Upload the workspace folder available in the GitHub repository to the Udacity's Workspace
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Open another terminal and type "env|grep WORK". You can see the space ID (it will start with view*** and some characters after that)

5. Open your web browser and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the letters after "view" with your space id that you got in the step 4. Press enter and you can see the app's output


## Acknowledgements:
Thanks to FigureEight for providing the dataset used in this project and, Udacity for guiding with the necessary steps for completing the project.


## Snapshots of the web app
![Project Snapshot 1](/workspace/project_snapshots/snapshot_1.png)
![Project Snapshot 2](/workspace/project_snapshots/snapshot_2.png)
![Project Snapshot 3](/workspace/project_snapshots/snapshot_3.png)
![Project Snapshot 4](/workspace/project_snapshots/snapshot_4.png)
![Project Snapshot 5](/workspace/project_snapshots/snapshot_5.png)