import pandas as pd
import numpy as np
from typing import Tuple
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

class WeatherPreprocessor:
    def __init__(self,num_cols):
        self.num_cols = num_cols
    
    def fix_weather_outliers(self,df: pd.DataFrame) -> pd.DataFrame:

        # Temperature 
        for c in ["temp", "temp_min", "temp_max"]:
            if c in df.columns:
                df[c] = df[c].clip(-40, 55)

        # Pressure
        if "pressure" in df.columns:
            df["pressure"] = df["pressure"].clip(870, 1085)

        # Humidity
        if "humidity" in df.columns:
            df["humidity"] = df["humidity"].clip(0, 100)

        # Wind
        if "wind_speed" in df.columns:
            df["wind_speed"] = df["wind_speed"].clip(
                df["wind_speed"].quantile(0.01),
                df["wind_speed"].quantile(0.99)
            )

        if "wind_deg" in df.columns:
            df["wind_deg"] = df["wind_deg"].clip(0, 360)

        # Rain / Snow (no negatives)
        for c in ["rain_1h", "rain_3h", "snow_3h"]:
            if c in df.columns:
                df[c] = df[c].clip(lower=0)

        # Clouds
        if "clouds_all" in df.columns:
            df["clouds_all"] = df["clouds_all"].clip(0, 100)

        log.info('Fixed weather outlier')

        return df

    def process_weather(self,df: pd.DataFrame) -> pd.DataFrame:
        num_cols = self.num_cols
        #### convert datetime col #########
        df.rename(columns={
        'dt_iso' : 'datetime'
        }, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        
        #### drop duplicate ####
        df.drop_duplicates(subset=['datetime', 'city_name'], keep='last',inplace=True)
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

        ### fill na for numeric col ###
        df[num_cols] = (
            df[num_cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
        )

        null_zero_cols = []
        for col in df.columns:
            if (col in num_cols) and (df[col] == 0.0).all():
                null_zero_cols.append(col)
            if col not in num_cols and df[col].isna().all():
                null_zero_cols.append(col)

        ### drop all zero cols ###
        df.drop(columns=null_zero_cols,inplace=True)
        log.info("Weather data cleaning complete")
        return df
    
    def split_city_wise_weather(self,df: pd.DataFrame) -> pd.DataFrame:
        df['city_name'] = df['city_name'].astype(str).str.strip().str.lower()

        valencia_df = df[df['city_name'] == 'valencia'].copy()
        madrid_df = df[df['city_name'] == 'madrid'].copy()
        bilbao_df = df[df['city_name'] == 'bilbao'].copy()
        barcelona_df = df[df['city_name'] == 'barcelona'].copy()
        seville_df = df[df['city_name'] == 'seville'].copy()

        city_df_list = [valencia_df, madrid_df, bilbao_df, barcelona_df, seville_df]

        for city_df in city_df_list:
            if city_df.empty:
                continue  

            non_weather_param = ['datetime', 'city_name']
            city_name = city_df['city_name'].iloc[0]

            city_df.columns = [
                col if col in non_weather_param else f"{city_name}_{col}"
                for col in city_df.columns
            ]
            city_df.drop(columns = ['city_name'],inplace=True)

            city_df["datetime"] = pd.to_datetime(city_df["datetime"])
            city_df.set_index("datetime", inplace=True)

        final_df = pd.concat(city_df_list, axis=1).reset_index()

        log.info("City wise weather processed")

        return final_df
    
    def prepare_weather(self, df):
        df = self.process_weather(df)
        df = self.fix_weather_outliers(df)
        return self.split_city_wise_weather(df)

class DemandPreprocessor:
    def __init__(self,num_cols):
        self.num_cols = num_cols
    
    def process_generation_demand(self,df: pd.DataFrame) -> pd.DataFrame:
        num_cols = self.num_cols
        #### convert datetime col #########
        df.rename(columns={
        'time' : 'datetime'
        }, inplace=True) 

        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')

        #### drop duplicate ####
        df.drop_duplicates(subset='datetime', inplace=True)
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

        ### fill na for numeric col ###
        df[num_cols] = (
            df[num_cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
        )
           
        df['total load actual'] = df['total load actual'].where(
                    df['total load actual'] != 0,
                    df['total load forecast']
                )

        null_zero_cols = []
        for col in num_cols:
            if col != 'datetime' and (df[col] == 0.0).all():
                null_zero_cols.append(col)
        
        for col in df.columns:
            if col not in num_cols and df[col].isna().all():
                null_zero_cols.append(col)


        ### drop all zero cols ###
        df.drop(columns=null_zero_cols,inplace=True)
        
        return df
    
class DataPreprocessor:
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load electricity total load actual and weather data"""
        df = pd.read_csv(file_path)
        return df
    

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_month'] = df['datetime'].dt.day
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        log.info('Created time based feature')
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'total load actual') -> pd.DataFrame:
        """Create lag features"""
        df = df.copy()
        df['lag1'] = df[target_col].shift(24)
        df['lag7'] = df[target_col].shift((24*7))

        df['lag1'] = df['lag1'].fillna(0)
        df['lag7'] = df['lag7'].fillna(0)
        log.info("Created Lag parameters")
        return df
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'total load actual') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Complete preprocessing pipeline"""
        df = self.create_time_features(df)
        df = self.create_lag_features(df, target_col)
        
        # Drop NaN values created by lag and rolling features
        df = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['datetime', target_col]]
        self.feature_names = feature_cols
        
        return df
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler"""
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = df.dropna()
        
        X = df[self.feature_names].values
        
        return X
    
class TimeTargetEncoder:

    def __init__(self, cols, time_col, n_splits=5, smoothing=10):
        """
        Args:
            cols: List of categorical columns to encode
            time_col: Name of the datetime column for time-based splitting
            n_splits: Number of time-based folds for encoding
            smoothing: Smoothing parameter for encoding (higher = more regularization)
        """
        self.cols = cols
        self.time_col = time_col
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.global_mean = None
        self.maps = {}
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the encoder on training data and transform it.
        Uses time-based cross-validation.
        """
        df = X.copy()
        df["target"] = y.values
        
        # Sort by time 
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        # Calculate global mean for smoothing and unseen categories
        self.global_mean = df["target"].mean()
        
        fold_size = len(df) // self.n_splits
        
        # Initialize encoded columns with global mean
        for col in self.cols:
            df[f"{col}_te"] = self.global_mean
        
        # Time-based encoding: use only past data to encode future data
        for i in range(1, self.n_splits):
            train_df = df.iloc[: i * fold_size]
            val_df = df.iloc[i * fold_size : (i + 1) * fold_size]
            
            for col in self.cols:
                # Calculate statistics only on training portion
                stats = train_df.groupby(col)["target"].agg(["mean", "count"])
                
                # Apply smoothing (Bayesian encoding)
                enc_map = (
                    (stats["count"] * stats["mean"] + self.smoothing * self.global_mean)
                    / (stats["count"] + self.smoothing)
                )
                
                # Apply encoding to validation fold
                df.loc[val_df.index, f"{col}_te"] = (
                    val_df[col].map(enc_map).fillna(self.global_mean)
                )
        
        # Handle the first fold (use global mean since no prior data)
        first_fold_idx = df.iloc[:fold_size].index
        for col in self.cols:
            df.loc[first_fold_idx, f"{col}_te"] = self.global_mean
        
        train_cutoff = int(len(df) * (self.n_splits - 1) / self.n_splits)
        final_train_df = df.iloc[:train_cutoff]
        
        for col in self.cols:
            stats = final_train_df.groupby(col)["target"].agg(["mean", "count"])
            self.maps[col] = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean)
                / (stats["count"] + self.smoothing)
            ).to_dict()
        
        return df.drop(columns=self.cols + ["target"])
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted encoding maps.
        """
        if self.global_mean is None:
            raise ValueError("Encoder has not been fitted. Call fit_transform first.")
        
        X = X.copy()
        
        for col in self.cols:
            X[f"{col}_te"] = X[col].map(self.maps[col]).fillna(self.global_mean)
        
        return X.drop(columns=self.cols)

class DataSplitter:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "total load actual",
        test_size: float = settings.TEST_SIZE
    ):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.test_size = test_size

    def _split(self) -> None:
        """Internal method to split train and test only once."""

        split_idx = int(len(self.df) * (1 - self.test_size))

        self.train_data = self.df.iloc[:split_idx].copy()
        self.test_data = self.df.iloc[split_idx:].copy()

        self._split_done = True

    def load_train_val(self) -> Tuple[pd.DataFrame, pd.Series]:
        self._split()

        X_train_val = self.train_data.drop(columns=[self.target_col])
        y_train_val = self.train_data[self.target_col]
        log.info("Training Data split Complete")

        return X_train_val, y_train_val

    def load_test(self,filter_date=None) -> Tuple[pd.DataFrame, pd.Series]:
        self._split()
        self.test_data['datetime'] = pd.to_datetime(self.test_data["datetime"], utc=True)
        if filter:
            filtered_test_data = self.test_data[
                                        self.test_data["datetime"].dt.normalize() == filter_date
                                    ]
        else:
            filtered_test_data = self.test_data

        X_test = filtered_test_data.drop(columns=[self.target_col])
        y_test = filtered_test_data[self.target_col]
        log.info("Prediction Data split Complete")

        return X_test, y_test

    
