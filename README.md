# Disaster Response Pipeline Project

### Summary:
This project is for categorizing the messages sent during disaster events so that the messages can be forwarded to an appropriate disaster relief agency.

The categorizing of messages are done using a machine learning pipeline. The categorizing of the messages can be seen on a web app. If someone wants to know the message category, they can type of the search field of the web app and then click on "Classify Message" button to see the classified category.

### Installations:


### Instructions:
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

![Project Snapshot 1](/workspace/project_snapshots/snapshot_1.png)
![Project Snapshot 2](/workspace/project_snapshots/snapshot_2.png)
![Project Snapshot 3](/workspace/project_snapshots/snapshot_3.png)
![Project Snapshot 4](/workspace/project_snapshots/snapshot_4.png)
![Project Snapshot 5](/workspace/project_snapshots/snapshot_5.png)