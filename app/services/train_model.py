from sklearn.metrics import mean_absolute_percentage_error
from app.services.preprocessing import (DataPreprocessor,
                                        DemandPreprocessor,
                                        WeatherPreprocessor,
                                        DataSplitter,
                                        TimeTargetEncoder)
from app.core.config import settings,model_config
from app.prediction_models.xgboost import XGBoostModel
import pandas as pd
import joblib
from app.core.logger import get_logger

log = get_logger(__name__)

def save_encoder(encoder,path:str):
    joblib.dump(encoder, path)
    log.info("Encoder Saved")
    
def load_encoder(path: str):
    encoder = joblib.load(path)
    log.info("Encoder Loaded")
    return encoder

def process_data():
    data_processor  = DataPreprocessor()
    demand_processor = DemandPreprocessor(num_cols=model_config.numeric_energy_df_col)
    weather_processor = WeatherPreprocessor(num_cols=model_config.numeric_weather_col)


    energy_df = data_processor.load_data(settings.ENERGY_DATA_PATH)
    processed_energy_df = demand_processor.process_generation_demand(energy_df)
    log.info('Demand Data Processed')

    weather_df = data_processor.load_data(settings.WEATHER_DATA_PATH)
    processed_weather_df = weather_processor.prepare_weather(weather_df)
    log.info('Weather Data Processed')

    processed_dataset = pd.merge(processed_energy_df,processed_weather_df, on='datetime', how='inner')

    processed_dataset = processed_dataset.drop(columns=model_config.non_target_col)

    final_df = data_processor.preprocess(processed_dataset,target_col='total load actual')
    log.info("Data Processing Complete")

    return final_df

    
def train_model(X_train_val, y_train_val):
    weather_cat_cols = [  
        col for col in X_train_val.columns 
        if any(org_col in col for org_col in model_config.org_weather_cat_cols)
    ]

    encoder = TimeTargetEncoder(
        cols=weather_cat_cols,
        time_col="datetime",
        n_splits=5,
        smoothing=10
    )

    xgb_model = XGBoostModel(encoder=encoder)
    
    xgb_model.train(
        X_train=X_train_val,
        y_train=y_train_val
    )
    
    save_encoder(encoder, settings.ENCODER_PATH)
    xgb_model.save(settings.TRAINED_MODEL_PATH)  


def predict_and_explain(X_test, y_test, model:str='xgboost'):
    encoder = load_encoder(settings.ENCODER_PATH)
    xgb_model = XGBoostModel(encoder=encoder)
    xgb_model.load(settings.TRAINED_MODEL_PATH)
    
    prediction, shap_value = xgb_model.predict(X_test, return_shap=True)
    mape = mean_absolute_percentage_error(y_test, prediction)*100
    
    return prediction, y_test, shap_value, mape