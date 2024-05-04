import argparse
import os
import tensorflow as tf
from TFClassifier.tf_train import train as tf_train
from TFClassifier.tf_eval import evaluate as tf_evaluate
from TorchClassifier.torch_train import train as torch_train
from TorchClassifier.torch_eval import evaluate as torch_evaluate
from DatasetTools.dataset_loader import load_data

def main(model_type, dataset_path):
    # Load your dataset
    train_data, test_data = load_data(dataset_path)

    if model_type.lower() == 'tensorflow':
        # TensorFlow Model Training and Evaluation
        print("Training TensorFlow model...")
        tf_model = tf_train(train_data)
        print("Evaluating TensorFlow model...")
        tf_metrics = tf_evaluate(tf_model, test_data)
        print(f"TensorFlow Evaluation Metrics: {tf_metrics}")
    elif model_type.lower() == 'pytorch':
        # PyTorch Model Training and Evaluation
        print("Training PyTorch model...")
        torch_model = torch_train(train_data)
        print("Evaluating PyTorch model...")
        torch_metrics = torch_evaluate(torch_model, test_data)
        print(f"PyTorch Evaluation Metrics: {torch_metrics}")
    else:
        print(f"Model type '{model_type}' not recognized. Please use 'tensorflow' or 'pytorch'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model on a specified dataset.')
    parser.add_argument('model_type', type=str, help='Type of the model to train: "tensorflow" or "pytorch"')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()

    main(args.model_type, args.dataset_path)
