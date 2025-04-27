import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from survival_model import __version__ as _version
from survival_model.config.core import config
from survival_model.pipeline import survival_pipe
from survival_model.processing.data_manager import load_pipeline
from survival_model.processing.data_manager import pre_pipeline_preparation
from survival_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
survival_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = survival_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = survival_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'age':[75], 'anaemia':[0], 'creatinine_phosphokinase':[582], 'diabetes':[0],
       'ejection_fraction':[20], 'high_blood_pressure':[1], 'platelets':[265000.00],
       'serum_creatinine':[1.9], 'serum_sodium':[130], 'sex':[1], 'smoking':[0], 'time':[4]}
    
    make_prediction(input_data=data_in)
