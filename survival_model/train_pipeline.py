import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from survival_model.config.core import config
from survival_model.pipeline import survival_pipe
from survival_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting on training set
    survival_pipe.fit(X_train,y_train)
    
    # Performance on test set
    y_pred = survival_pipe.predict(X_test)
    print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= survival_pipe)
    # printing the score
    
if __name__ == "__main__":

    # import mlflow
    # import os
    # mlflow.set_tracking_uri("http://43.204.236.72:5000/")
    # mlflow.set_experiment("survival Survival Prediction")
    # mlflow.sklearn.autolog()

    # mlflow.start_run(run_name = os.environ['GIT_COMMIT_MSG'])
    
    # your training code goes here
    run_training()
    
    # mlflow.end_run()
