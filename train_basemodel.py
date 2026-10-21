"""
This script is designed to train an individual base model. 
Parameters:
    -o, --output_path: Path to save the model.
    -c, --config_path: Path to the configuration file (json).
Usage example:
    python3 train_basemodel.py -o models/mini/model1/
"""
import os 
import argparse
import shutil
from src.utils import load_config
from src.train import train

def parser():
    parser = argparse.ArgumentParser(description="Train a base model.")
    parser.add_argument("-o", "--output_path", type=str, required=True, 
                        help="Path to save the model.")    
    parser.add_argument("-c", "--config_path", type=str, required=False, 
                        help="Path to the configuration file (JSON).",
                        default="config/base.json")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args

if __name__ == "__main__":
    args = parser()
    config_path = args.config_path
    output_path = args.output_path

    # Copy the config file to the output path
    shutil.copyfile(config_path, os.path.join(output_path, "config.json"))

    # Load the configuration
    config = load_config(config_path)
    cat_file = os.path.join(config['data_path'], "categories.txt")
    categories = [line.strip() for line in open(cat_file, "r", encoding="utf-8")]

    # Train the model 
    print("Training the model...")
    train(config, categories, output_path)