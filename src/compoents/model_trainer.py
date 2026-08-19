import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


from src.exception import CoustomException
from src.logger import logging
from src.utils import evaluate_model,save_object


@dataclass
class modeltrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class modeltrainer:
    def __init__(self):
        self.model_trainer_config=modeltrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('splitting training and test data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models ={
                        'linear_regression':LinearRegression(),
                        'lasso':Lasso(),
                        'Ridge':Ridge(),
                        'KNeighborsRegressor':KNeighborsRegressor(),
                        'DecisionTreeRegressor':DecisionTreeRegressor(),
                        'AdaBoostRegressor': AdaBoostRegressor(),
                        'RandomForestRegressor': RandomForestRegressor(),
                        'XGBRegressor': XGBRegressor(),
                        'Catboost':CatBoostRegressor(verbose=False)}
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models)
            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CoustomException('No best model found')
            logging.info('Best model found for both training and test datasets')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            predicted = best_model.predict(X_test)
            r2_square= r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CoustomException(e,sys)