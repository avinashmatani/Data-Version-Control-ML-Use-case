from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
from sklearn.linear_model import ElasticNet
import os
import joblib

def train(config_path, config_params):

    config = read_yaml(config_path)
    params = read_yaml(config_params)
    
    # save data in local dir
    # creating path to directory: artifacts/raw_local_dir/data.csv

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config['artifacts']['split_data_dir']
    
    train_data_filename = config['artifacts']['train']

    train_data_path = os.path.join(artifacts_dir,split_data_dir,train_data_filename)

    train_data = pd.read_csv(train_data_path)
    
    train_y = train_data['quality']
    train_x = train_data.drop('quality', axis= 1)

    alpha = params['model_params']['ElasticNet']['params']['alpha']
    l1_ratio = params['model_params']['ElasticNet']['params']['l1_ratio']
    random_state = params['base']['random_state']

    lr = ElasticNet(alpha = alpha, l1_ratio= l1_ratio, random_state= random_state)
    lr.fit(train_x, train_y)


    model_dir = config['artifacts']['model_dir']
    model_filename = config['artifacts']['model_filename']
    
    create_directory([os.path.join(artifacts_dir,model_dir)])

    model_path = os.path.join(artifacts_dir,model_dir, model_filename)

    joblib.dump(lr, model_path)

    print("model training done")


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config" , "-c", default="config/config.yaml")
    args.add_argument("--params" , "-p", default="params.yaml")

    parsed_args = args.parse_args()
    print(parsed_args.params)

    train(config_path= parsed_args.config, config_params= parsed_args.params)



