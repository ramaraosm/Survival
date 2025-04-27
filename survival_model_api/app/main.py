import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
import numpy as np
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
from survival_model.processing.data_manager import load_dataset, load_pipeline
from survival_model import __version__ as _version
from survival_model.config.core import config
from sklearn.model_selection import train_test_split
from survival_model.predict import make_prediction

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# FastAPI object
app = FastAPI()


# UI - Input components
# Inputs from user
# 	age	anaemia	creatinine_phosphokinase	diabetes	ejection_fraction	high_blood_pressure	platelets	
# serum_creatinine	serum_sodium	sex	smoking	time	DEATH_EVENT
#	75.0	0	582	0	20	1	265000.00	1.9	130	1	0	4	1
in_age = gradio.Number(value="75", label='Age')
in_anaemia = gradio.Number(value="0", label='Anaemia')
in_creatinine = gradio.Number(value="582", label='Creatinine')
in_diabetes = gradio.Number(value="0", label='Diabetes')
in_ef = gradio.Number(value="20", label='Ejection fraction')
in_hbp = gradio.Number(value="1", label='high_blood_pressure')
in_platelets = gradio.Number(value="265000.00", label='platelets')
in_sc = gradio.Number(value="1.9", label='Serum creatinine')
in_ss = gradio.Number(value="130", label='Serum sodium')
in_sex = gradio.Number(value="1", label='Sex')
in_smoking = gradio.Number(value="0", label='Smoking')
in_time = gradio.Number(value="4", label='Time')

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

def predict_death_event(in_age,in_anaemia,in_creatinine,in_diabetes,in_ef,in_hbp,in_platelets,in_sc,in_ss,in_sex,in_smoking,in_time):
    #input_features = [in_age, in_anaemia, in_creatinine, in_diabetes, in_ef, in_hbp, in_platelets, in_sc, in_ss, in_sex, in_smoking, in_time]
    #input_array = np.array(input_features, dtype=np.float32).reshape(1, -1)
    #prediction = model.predict(input_array)
    input_df = pd.DataFrame({"age": [float(in_age)], 
                             "anaemia": [int(in_anaemia)], 
                             "creatinine_phosphokinase": [int(in_creatinine)],
                             "diabetes": [int(in_diabetes)], 
                             "ejection_fraction": [int(in_ef)], 
                             "high_blood_pressure": [int(in_hbp)],
                             "platelets": [float(in_platelets)], 
                             "serum_creatinine": [float(in_sc)], 
                             "serum_sodium": [int(in_ss)],
                             "sex": [int(in_sex)], 
                             "smoking": [int(in_smoking)],
                             "time": [int(in_time)]
                             })
    
    result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
    label = "Not Survived" if result[0]==1 else "Survived"
    return label

# Create Gradio interface object
iface = gradio.Interface(fn = predict_death_event,
                         inputs = [in_age,in_anaemia,in_creatinine,in_diabetes,in_ef,in_hbp,in_platelets,in_sc,in_ss,in_sex,in_smoking,in_time],
                         outputs = [out_label],
                         title="survival Survival Prediction API  ⛴",
                         description="Predictive model that answers the question: “What sort of people were more likely to survive?”",
                         allow_flagging='never'
                         )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
