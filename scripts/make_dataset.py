import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from constants import BATCH_SIZE, TOKENIZER_MAX_LENGTH


def get_data_loaders(texts):
    """
    Load and preprocess the MRPC dataset and return the training and validation dataloaders
    :param tokenizer: The tokenizer to use for preprocessing the dataset
    :return: The training and validation dataloaders
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # # Load and preprocess the MRPC dataset
    # dataset = load_dataset("glue", "mrpc")
    # texts = [
    #     (
    #         tokenizer(
    #             example["sentence1"],
    #             example["sentence2"],
    #             truncation=True,
    #             padding="max_length",
    #             max_length=TOKENIZER_MAX_LENGTH
    #         ),
    #         example["label"]
    #     )
    #     for example in dataset["train"]
    # ]
    input_ids = torch.tensor([t[0]["input_ids"] for t in texts])
    attention_masks = torch.tensor([t[0]["attention_mask"] for t in texts])
    labels = torch.tensor([t[1] for t in texts])

    # Split the dataset into training and validation sets
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids,
        labels,
        random_state=0,
        test_size=0.1
    )
    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks,
        labels,
        random_state=0,
        test_size=0.1
    )

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create the DataLoader for our training set
    train_data = TensorDataset(
        train_inputs,
        train_masks,
        train_labels
    )
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # Create the DataLoader for our validation set
    validation_data = TensorDataset(
        validation_inputs,
        validation_masks,
        validation_labels
    )
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)

    return train_dataloader, validation_dataloader
