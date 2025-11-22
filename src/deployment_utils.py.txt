## 3. `src/deployment_utils.py`

from huggingface_hub import hf_hub_download
import xgboost as xgb
import pandas as pd

def load_model_from_hf():
    local_path = hf_hub_download(
        repo_id='pallabbh/tourism-model',
        filename='best_xgb_model.json'
    )
    model = xgb.XGBClassifier()
    model.load_model(local_path)
    return model

def predict(input_dict):
    model = load_model_from_hf()
    df = pd.DataFrame([input_dict])
    # One-hot/label encoding as per training
    return model.predict(df)[0]
