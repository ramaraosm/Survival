import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from survival_model.config.core import config
from survival_model.processing.features import OutlierHandler

survival_pipe = Pipeline([

    ('handle_outliers_creatinine_phosphokinase', OutlierHandler(variable = config.model_config.creatinine_phosphokinase_var)),
    ('handle_outliers_ejection_fraction', OutlierHandler(variable = config.model_config.ejection_fraction_var)),
    ('handle_outliers_platelets', OutlierHandler(variable = config.model_config.platelets_var)),
    ('handle_outliers_serum_creatinine', OutlierHandler(variable = config.model_config.serum_creatinine_var)),
    
    # fearure scaling
    ("scaler", StandardScaler()),
    
    # ML model
    ('model_rf', XGBClassifier(n_estimators = config.model_config.n_estimators,
                                        max_depth = config.model_config.max_depth,
                                        random_state = config.model_config.random_state,
                                        max_leaves = config.model_config.max_leaves))
    
    ])