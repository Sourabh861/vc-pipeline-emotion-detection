import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import yaml

import logging

# logging configure

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)

        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size

    except FileNotFoundError:
        logger.error('File not found')
        raise

    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise

    except Exception as e:
        logger.error('some error occurred')
        raise 

def read_data(url: str) -> pd.DataFrame:

    df = pd.read_csv(url)
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:

    df.drop(columns=['tweet_id'], inplace=True)

    final_df = df[df['sentiment'].isin(['sadness', 'neutral'])]

    final_df['sentiment'].replace({'neutral':1,'sadness':0}, inplace=True)

    return final_df

def save_data(data_path: str,train_data: pd.DataFrame,test_data: pd.DataFrame) -> None:

    os.makedirs(data_path)

    train_data.to_csv(os.path.join(data_path,"train.csv"))
    test_data.to_csv(os.path.join(data_path,"test.csv"))

def main():

    test_size = load_params('params.yaml')
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    
    final_df = process_data(df)

    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    data_path = os.path.join("data","raw")

    save_data(data_path, train_data, test_data)

if __name__ == "__main__":

    main()