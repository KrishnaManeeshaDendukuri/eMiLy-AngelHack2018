import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
#from handler import target_column


def make_train_test_split(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    return X_train,X_test,y_train,y_test

def sample_randomly(range,sample_int=False,use_log_scale=False,from_list=False,volume=0):
    sample_array=[]
    if sample_int==True and use_log_scale==False and from_list==False:
        lower_val, upper_val = range
        tracker=0

        while tracker < volume:

            sample=np.random.randint(low=lower_val,high=upper_val+1)
            if sample not in sample_array:
                sample_array.append(sample)
                tracker=tracker+1
        return sample_array

    if sample_int==False and use_log_scale==True and from_list==False:


        lower_val,upper_val=range
        lower_val=np.log10(lower_val)
        upper_val=np.log10(upper_val)
        tracker = 0
        while tracker<volume:
            sample=random.uniform(lower_val,upper_val)
            sample=10**sample
            if sample not  in sample_array:
                sample_array.append(sample)
                tracker = tracker + 1
        return sample_array

    if sample_int==False and from_list==True:
        tracker=0
        while tracker<volume:
            sample=np.random.randint(low=0,high=len(range))
            sample_array.append(range[sample])
            tracker = tracker + 1
        return sample_array



def create_parameter_grid(param_list,specification_list,range_list,sample_volume):
    param_grid={}
    
    """
    for param in param_list:
        param_grid[param]=[]
    """

    for i in range(len(param_list)):
        param_grid[param_list[i]]=sample_randomly(range=range_list[i],sample_int=specification_list[i][0],use_log_scale=specification_list[i][1],from_list=specification_list[i][2],volume=sample_volume)
    return param_grid


def combine_pred_and_test(test,pred,model_name,target_column):
    predicted_column="predicted_"+target_column
    test[predicted_column]=pred
    output_dir="output/"+str(model_name)+"_prediction.csv"
    test.to_csv(output_dir)

