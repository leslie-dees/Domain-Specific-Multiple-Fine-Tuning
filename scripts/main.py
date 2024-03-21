import torch
from transformers import BertTokenizer
import pandas as pd
# Local imports
from constants import TOKENIZER_MAX_LENGTH
from datasets import load_dataset
from make_dataset import get_data_loaders
from model import get_base_model
from setup import training_model

# Define example sentences to use for prediction before and after fine-tuning
EXAMPLE_SENTENCE_1 = "The company reported better than expected results."
EXAMPLE_SENTENCE_2 = "The firm's results exceeded forecasts."
EXAMPLE_SENTENCE_1_MED = "The patient now presents with a three to four week history of shortness of breath and a dry non-productive cough."
EXAMPLE_SENTENCE_2_MED = "The patient now gives a three to four week history of shortness of breath and a dry non-gainful wheeze."

def main():
    """
    Main function to run the fine-tuning process for a BERT model on a specified dataset.
    It evaluates the model performance before and after fine-tuning.
    """

    # Set up the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)
    
    # Initialize the tokenizer from the pre-trained BERT model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print('tokenizer done')
    # Direct Fine Tuning: Approach 1
    # Load the data
    # Load and preprocess the MRPC dataset
    df = pd.read_csv('data/processed/medicalCorpus.csv')
    print(df)
    # Tokenize sentences
    texts = [(tokenizer(example['Sentence_1'], example['Sentence_2'], truncation=True, padding='max_length', max_length=128), example['Paraphrase_Indicator']) for index, example in df.iterrows()]
    train_loader, validation_loader = get_data_loaders(texts)
    print('created loaders')
    model = get_base_model()
    training_model(device, tokenizer, train_loader, validation_loader, model, EXAMPLE_SENTENCE_1_MED, EXAMPLE_SENTENCE_2_MED, save_file='med_direct')

    # Multiple Fine Tuning: Approach 2
    # Step1: Finetuning MRPC dataset
    dataset = load_dataset("glue", "mrpc")
    texts = [
            (
                tokenizer(
                    example["sentence1"],
                    example["sentence2"],
                    truncation=True,
                    padding="max_length",
                    max_length=TOKENIZER_MAX_LENGTH
                ),
                example["label"]
            )
            for example in dataset["train"]
        ]
    train_loader, validation_loader = get_data_loaders(texts)
    model = get_base_model()
    training_model(device, tokenizer, train_loader, validation_loader, model, EXAMPLE_SENTENCE_1, EXAMPLE_SENTENCE_2, save_file='mrpc')


    model = get_base_model()
    # Load the state dictionary
    state_dict = torch.load('models/model_fine_tuned_mrpc.pt')
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    training_model(device, tokenizer, train_loader, validation_loader, model, EXAMPLE_SENTENCE_1_MED, EXAMPLE_SENTENCE_2_MED, save_file='med_on_mrpc')


# Check if this script is executed as the main program and run the main function
if __name__ == "__main__":
    main()
