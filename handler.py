from model import Models
from utils import make_train_test_split
import pandas as pd
import os

filename = "LoanPrediction.csv"           #change filename
dname = os.getcwd() + '/Dataset'
filename = os.path.join(dname,filename)
data=pd.read_csv(filename)

target_column = "Loan_Status"                         #change target_column
y = data[target_column][0:]

cols=list(data.columns)
cols.remove(target_column)
X = data[cols][0:]

X_train,X_test,y_train,y_test = make_train_test_split(X,y)
model = Models(X_train,X_test,y_train,y_test)


#print(model.VanillaLinearRegression())
#print(model.LassoLinearRegression())
#print(model.RidgeLinearRegression())
#print(model.RandomForestRegressor())
#print(model.GradientBoostedRegressor())
#print(model.SupportVectorRegressor())

print(model.VanillaLogisticRegression())
print(model.RandomForestClassifier())
print(model.GradientBoostedClassifier())
print(model.SupportVectorClassifier())

#model.XtremeGradientBoosting()

print(filename)
