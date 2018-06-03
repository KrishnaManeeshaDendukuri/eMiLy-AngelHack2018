import pandas as pd
import numpy as np
import os
import pathlib

def input_file(path):
        
    all_files = []
    
    os.chdir("Dataset")
    
    for file in pathlib.Path('.').glob("*.csv"):  
        
        if "Book1.csv" in str(file.absolute()):
                continue
            
        df = pd.read_csv(os.path.join(os.getcwd(),file)).values.tolist()
        try:
            padd_row_count = 1024 - len(df)
            if padd_row_count < 0:
                raise Exception
        except Exception as e:
            padd_row_count = 0
            df = pd.read_csv(os.path.join(os.getcwd(),file))
            df = df[:1024].values.tolist()
        padd_col_count = 32 - len(df[0])
        df = np.pad(np.array(df), [(0, padd_row_count), (0, padd_col_count)], mode='constant', constant_values=0)
        all_files.append(df)
    all_file = np.array(all_files)
    
    
    Y_label_header = []
    Y_label = []
    Y = []
    
    os.chdir(os.path.join(os.getcwd(),"_Y"))
    for file in pathlib.Path('.').glob("*.csv"):
        
        df = pd.read_csv(os.path.join(os.getcwd(),file))
        df = df["entry"]
        header = df[0]
        df = df[1:].values.tolist()
        Y_label_header.append(header)
        Y.append(df)
    
    Y = np.array(Y)
    
    os.chdir("..")
    
    for ind,file in enumerate(pathlib.Path('.').glob("*.csv")):  
        
        if "Book1.csv" in str(file.absolute()):
                continue
            
        df = pd.read_csv(os.path.join(os.getcwd(),file))
        df = df[Y_label_header[ind-1]].values.tolist()
        
        padd_col_count = 0
        try:
            padd_row_count = 1024 - len(df)
            if padd_row_count < 0:
                raise Exception
        except Exception as e:
            padd_row_count = 0
            df = pd.read_csv(os.path.join(os.getcwd(),file))
            df = df[Y_label_header[ind-1]]
            df = df[:1024].values.tolist()
        
        df = np.pad(np.array(df),(0,padd_row_count),'constant')       
        Y_label.append(df)   
        
    Y_label = np.array(Y_label)
    
    os.chdir("..")
    
    return all_file, Y_label, Y
    

#a, Y_label, Y = input_file(os.getcwd())

