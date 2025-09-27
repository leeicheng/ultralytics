import argparse
import yaml
from ultralytics import YOLO

# Import your custom components to ensure they are registered
import courtpoints

def train(args):
    """Starts the training process using a configuration file."""
    print(f"\nStarting Court Points Training from config: {args.config}")

    # Load the training arguments from the YAML file
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)

    # Extract arguments for the train function
    # The 'data' argument for model.train() should be the config file itself
    # The framework will read path, train, val from it.
    train_params = {
        'data': args.config, 
        'epochs': config_args.get('epochs', 100),
        'batch': config_args.get('batch', 16),
        'imgsz': config_args.get('imgsz', 640),
    }

    # The base model for transfer learning is also specified in the config
    model_to_load = config_args.get('model', 'yolov8n.pt')
    print(f"Loading base model: {model_to_load} for transfer learning...")
    model = YOLO(model_to_load)

    print(f"Starting training with parameters: {train_params}")
    results = model.train(**train_params)
    
    print("\nTraining finished.")
    print(f"Best model saved to: {results.save_dir}")

def predict(args):
    """Runs prediction using a trained CourtPoints model."""
    print("\nStarting Court Points Prediction...")
    
    model = YOLO(args.weights)
    
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save
    )
    
    print(f"\nPrediction finished. Results can be found in: {results[0].save_dir if args.save else 'Not saved (use --save flag)'}")

def main():
    parser = argparse.ArgumentParser(description="Main script for CourtPoints Detection tasks.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands: train, predict')

    # --- Train sub-command ---
    parser_train = subparsers.add_parser('train', help='Train a model using a configuration file.')
    parser_train.add_argument('--config', type=str, default='court_points_config.yaml', help='Path to the training configuration YAML file.')
    parser_train.set_defaults(func=train)

    # --- Predict sub-command ---
    parser_predict = subparsers.add_parser('predict', help='Run prediction with a trained CourtPoints model.')
    parser_predict.add_argument('--weights', type=str, required=True, help='Path to your trained model weights (.pt file).')
    parser_predict.add_argument('--source', type=str, required=True, help='Path to the source image or video file.')
    parser_predict.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions.')
    parser_predict.add_argument('--save', action='store_true', help='Save the prediction results (images/videos).')
    parser_predict.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()