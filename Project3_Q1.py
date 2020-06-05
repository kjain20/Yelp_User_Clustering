import pandas as pd
import numpy as np
import time

start_time = time.time()

def transform_data(filename):
    df = pd.read_csv("yelp.csv")
    
    df2 = df.copy()
    df2 = df2.drop(columns = ['name','user_id'])
    df2['yelping_since'] = pd.to_numeric(df2['yelping_since'].str[0:4], errors='coerce')
    df2= df2.replace('None',-1)
    df2['elite'] = df2['elite'].str.count(',')
    df2 = df2.fillna(-1)
    df2['elite'] = df2['elite'] + 1
    
    return np.array(df2)
