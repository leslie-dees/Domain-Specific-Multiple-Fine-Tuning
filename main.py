import torch
from transformers import BertTokenizer

# Local imports
from constants import EPOCHS
from scripts.make_dataset import get_data_loaders
from scripts.model import get_base_model, predict_on_example
from setup import fine_tune_model_on_data_loaders
from util import evaluate_model, save_model

# Define example sentences to use for prediction before and after fine-tuning
EXAMPLE_SENTENCE_1 = "The company reported better than expected results."
EXAMPLE_SENTENCE_2 = "The firm's results exceeded forecasts."

def main():
    """
    Main function to run the fine-tuning process for a BERT model on a specified dataset.
    It evaluates the model performance before and after fine-tuning.
    """
    # Set up the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the tokenizer from the pre-trained BERT model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load the data
    train_loader, validation_loader = get_data_loaders()  # Get DataLoader objects for train and validation sets
    
    # Load the base (pre-trained but not fine-tuned) model
    model_without_fine_tuning = get_base_model()
    save_model(model_without_fine_tuning, "model.pt")  # Save the base model for comparison
    
    # Fine-tune the model on the training data
    model_after_fine_tuning = fine_tune_model_on_data_loaders(
        model_without_fine_tuning,
        train_loader,
        device,
        epochs=EPOCHS
    )
    save_model(model_after_fine_tuning, "model_fine_tuned.pt")  # Save the fine-tuned model
    
    # Evaluate the model before fine-tuning using the example sentences
    probabilities_without_fine_tuning, prediction_without_fine_tuning = predict_on_example(
        model_without_fine_tuning,
        tokenizer,
        EXAMPLE_SENTENCE_1,
        EXAMPLE_SENTENCE_2,
        device
    )
    print(
        f"Prediction before fine-tuning: {prediction_without_fine_tuning}\nProbabilities: {probabilities_without_fine_tuning}"
    )
    
    # Calculate the accuracy of the model before fine-tuning on the validation set
    pre_fine_tune_accuracy = evaluate_model(
        model_without_fine_tuning,
        validation_loader,
        device
    )
    print(f'Accuracy before fine-tuning: {pre_fine_tune_accuracy:.4f}')

    # Evaluate the model after fine-tuning using the same example sentences
    probabilities_with_fine_tuning, prediction_with_fine_tuning = predict_on_example(
        model_after_fine_tuning,
        tokenizer,
        EXAMPLE_SENTENCE_1,
        EXAMPLE_SENTENCE_2,
        device
    )
    print(
        f"Prediction after fine-tuning: {prediction_with_fine_tuning}\nProbabilities: {probabilities_with_fine_tuning}"
    )
    
    # Calculate the accuracy of the fine-tuned model on the validation set
    post_fine_tune_accuracy = evaluate_model(
        model_after_fine_tuning,
        validation_loader,
        device
    )
    print(f'Accuracy after fine-tuning: {post_fine_tune_accuracy:.4f}')

# Check if this script is executed as the main program and run the main function
if __name__ == "__main__":
    main()
