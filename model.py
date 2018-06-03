import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error,accuracy_score,mean_absolute_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils import create_parameter_grid,combine_pred_and_test
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.svm import SVR,SVC
#import xgboost as xgb

class Models:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

    def VanillaLinearRegression(self):
        lin_model=linear_model.LinearRegression()
        lin_model.fit(self.X_train, self.y_train)
        predictions = lin_model.predict(self.X_test)
        error = accuracy_score(self.y_test, predictions)
        return 100-error
    
    def VanillaLogisticRegression(self):
        lin_model=linear_model.LogisticRegression()
        lin_model.fit(self.X_train, self.y_train)
        predictions = lin_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        return acc

    def LassoLinearRegression(self):
        est = linear_model.Lasso()
        #param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}
        param_grid=create_parameter_grid(["alpha"],[[False, True, False]],[(0.0001,1)],11)
        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=3,verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        error = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        return 100-error

    def RidgeLinearRegression(self):
        est = linear_model.Ridge()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}
        param_grid = create_parameter_grid(["alpha"], [[False, True, False]], [(0.0001, 1)], 11)
        est_model =GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=3,verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        error = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        return 100-error

    def RandomForestRegressor(self):
        est = RandomForestRegressor()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

        param_grid =create_parameter_grid(["n_estimators","max_depth"], [[True, False, False],[True, False, False]], [(50, 500),(2,15)], 2)

        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2,verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        error = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        return 100-error
    
    def RandomForestClassifier(self):
        est = RandomForestClassifier()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

        param_grid =create_parameter_grid(["n_estimators","max_depth"], [[True, False, False],[True, False, False]], [(50, 500),(2,15)], 2)

        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2,verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        return acc

    def GradientBoostedRegressor(self):
        est = GradientBoostingRegressor()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

        param_grid =create_parameter_grid(["n_estimators","max_depth","learning_rate"], [[True, False, False],[True, False, False],[False, True, False]], [(50, 500),(2,30),(0.000001,1)], 2)
        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2,verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        error = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        return 100-error
    
    def GradientBoostedClassifier(self):
        est = GradientBoostingClassifier()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

        param_grid =create_parameter_grid(["n_estimators","max_depth","learning_rate"], [[True, False, False],[True, False, False],[False, True, False]], [(50, 500),(2,30),(0.000001,1)], 2)
        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2,verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        return acc
    
    def SupportVectorRegressor(self):
        est = SVR()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

        param_grid = create_parameter_grid(["C", "epsilon"],
                                           [[False, True, False],[False, True, False]],
                                           [(0.1,5), (0.01, 1)], 5)

        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2, verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        error = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        return 100-error
    
    def SupportVectorClassifier(self):
        est = SVC()
        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

        param_grid = create_parameter_grid(["C"],
                                           [[False, True, False]],
                                           [(0.1,5)], 3)

        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2, verbose=0,
                                 scoring="neg_mean_squared_error")

        est_model.fit(self.X_train, self.y_train)
        predictions = est_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        return acc
    
    
    #    def XtremeGradientBoosting(self):
#        est = xgb.XGBRegressor()
#        # param_grid = {'alpha':np.arange(.000001,1,)}  # 'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}
#
#        param_grid =create_parameter_grid(["n_estimators","max_depth"], [[True, False, False],[True, False, False]], [(50, 500),(2,7)], 2)
#
#        print(param_grid)
#
#        est_model = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, cv=2,verbose=3,
#                                 scoring="neg_mean_squared_error")
#
#        est_model.fit(self.X_train, self.y_train)
#        best_params = est_model.best_params_
#        print(best_params)
#        predictions = est_model.predict(self.X_test)
#        error = np.sqrt(mean_squared_error(self.y_test, predictions))
#        print(error)
#        target_column = self.y_test.name
#        test_to_combine = self.X_test.copy(deep=True)
#
#        combine_pred_and_test(test_to_combine, predictions, "XtremeGradientBoosting",target_column)
#        return predictions, error, best_params




