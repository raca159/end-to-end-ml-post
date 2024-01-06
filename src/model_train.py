import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def get_data(args):
    '''
    Here data gets loaded and splitted into training and testing.
    It's kept as DataFrame to facilitate column based transformation in data later.
    '''
    df = pd.read_csv(args.input_data)
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0']
    Xdf = df.drop(columns=['output'])
    y = df['output'].values
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        np.arange(df.shape[0]), y, random_state=0, test_size=float(args.test_size)
    )
    return Xdf.iloc[X_train], Xdf.iloc[X_test], y_train, y_test

def train_model(args, X_train, y_train):
    '''
    For the model, we choose a random forest and we focus on the data transformation pipeline.
    By using the Pipeline object from scikit learn, we don't have to worry about writing transformation
    logic in the inference script, the model itself will handle the transformations.

    In this case we normalize numeric coluns such as age and etc, and apply a one-hot encoding for all
    "multi-class" columns, these columns hold "types" of data, so we convert these types into binary
    encoded columns to facilitate the models understanding of the impact of each "type" of each feature.

    Since the azure sweep don't allow for None parameters, we converted the `max_depth` and `max_features`
    None values into string when we launched the script, so we converted it back to None here.
    '''
    data_transformation_step = ColumnTransformer([
        ('std_scaler', StandardScaler(), ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']),
        ('one_hot_encoding', OneHotEncoder(), ['cp', 'restecg'])
    ])

    max_depth = args.max_depth
    if max_depth == -1:
        max_depth = None

    max_features = args.max_features
    if max_features == 'None':
        max_features = None

    model = Pipeline([
        ('transform_columns', data_transformation_step),
        ('random_forest', RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            criterion=args.criterion,
            max_features=max_features
        ))
    ])
    model = model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    '''
    With the model properly trained, we can now score it and acquire the accuracy of it, we could choose other metrics
    for evaluating a classification task and add them to the mlflow log. Given the simplicity of this dataset we
    will be measuring it via accuracy.
    '''
    # model accuracy for X_test
    score = model.score(X_test, y_test)
    return score

def save_model(args, model):
    # acquire model name in args
    model_name = args.model_name

    # Registering the model to the workspace
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=model_name,
        artifact_path=model_name
    )
    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(model_name, "trained_model"),
    )

def execution(args):
    # Start Logging
    mlflow.start_run()
    # enable autologging
    mlflow.sklearn.autolog()
    # record metrics
    mlflow.log_param('criterion', str(args.criterion))
    mlflow.log_param('max_features', str(args.max_features))
    mlflow.log_metric('n_estimators', float(args.n_estimators))
    mlflow.log_metric('max_depth', float(args.max_depth))

    # get data
    X_train, X_test, y_train, y_test = get_data(args)
    # train model
    model = train_model(args, X_train, y_train)
    # test model
    score = test_model(model, X_test, y_test)
    # save model
    save_model(args, model)
    # record results
    mlflow.log_metric('Score', float(score))
    # finish mlflow recording
    mlflow.end_run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data', type=str, help='Path for data'
    )
    parser.add_argument(
        '--model_name', type=str, default="test_model",
        help='Model name to be registered later'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.3,
        help='Test size for training and validation step'
    )
    parser.add_argument(
        '--n_estimators', type=int, default=100,
        help='n_estimators parameter of RandomForestClassifier model'
    )
    parser.add_argument(
        '--max_depth', type=int, default=-1,
        help='max_depth parameter of RandomForestClassifier model'
    )
    parser.add_argument(
        '--criterion', type=str, default='gini',
        help='criterion parameter of RandomForestClassifier model'
    )
    parser.add_argument(
        '--max_features', type=str, default='sqrt',
        help='max_features parameter of RandomForestClassifier model'
    )
    execution(parser.parse_args())

if __name__ == '__main__':
    main()