import pandas as pd


def  load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df 


def save_datest(df, save_path):
    df.to_csv(save_path)
    
    
