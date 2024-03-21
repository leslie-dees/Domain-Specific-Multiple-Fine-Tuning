import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from util import save_model, full_model_evaluation
from scripts.model import predict_on_example
from constants import EPSILON, LEARNING_RATE, WARMUP_STEPS, EPOCHS


def fine_tune_model_on_data_loaders(model, train_dataloader, device, epochs=5):
    """
    Fine-tunes a pre-trained BERT model on a given dataset.

    Parameters:
        model (torch.nn.Module): The pre-trained BERT model for sequence classification.
        train_dataloader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
        device (torch.device): The device to train the model on (CPU or GPU).
        epochs (int): Number of epochs to train the model.

    Returns:
        torch.nn.Module: The fine-tuned BERT model.
    """
    # Freeze all parameters of the model to prevent them from being updated during training
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the last three layers to allow updating during training
    for layer in [model.bert.encoder.layer[-1], model.bert.encoder.layer[-2], model.bert.encoder.layer[-3]]:
        for param in layer.parameters():
            param.requires_grad = True

    # Initialize the optimizer with only the unfrozen parameters (i.e., last three layers)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        eps=EPSILON
    )

    # Calculate the total number of training steps and set up the learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Switch the model to training mode
    model.train()
    # Iterate over each epoch
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0  # Track the total loss for each epoch
        # Iterate over each batch in the training data
        for batch in train_dataloader:
            # Move the batch tensors to the specified device
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from the dataloader
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()  # Reset gradients before performing backpropagation

            # Forward pass: compute the model output (logits)
            logits = model(b_input_ids, attention_mask=b_input_mask)
            # Compute the loss between model predictions and actual labels
            loss = nn.CrossEntropyLoss()(logits, b_labels)

            total_loss += loss.item()  # Accumulate the loss
            loss.backward()  # Perform backpropagation
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update the learning rate

        # Calculate the average loss over all batches for the current epoch
        avg_train_loss = total_loss / len(train_dataloader)
        # Print the average training loss for the epoch
        print(f"Average training loss: {avg_train_loss:.4f}")

    # Print completion message once fine-tuning is finished
    print("Finished fine-tuning.")
    return model  # Return the fine-tuned model


def training_model(device, tokenizer, train_loader, validation_loader, model, EXAMPLE_SENTENCE_1, EXAMPLE_SENTENCE_2, save_file):
     # Load the base (pre-trained but not fine-tuned) model
    model_without_fine_tuning = model
    save_model(model_without_fine_tuning, f"pre_trained_model_{save_file}.pt")  # Save the base model for comparison
    
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
    
    # Evaluate of the model before fine-tuning on the validation set
    pre_fine_tune_metrics = full_model_evaluation(
        model_without_fine_tuning,
        validation_loader,
        device
    )
    print("Metrics before fine-tuning")
    for item in pre_fine_tune_metrics:
        print(f"{item}: ", pre_fine_tune_metrics[item])
        
    # Fine-tune the model on the training data
    model_after_fine_tuning = fine_tune_model_on_data_loaders(
        model_without_fine_tuning,
        train_loader,
        device,
        epochs=EPOCHS
    )
    save_model(model_after_fine_tuning, f"model_fine_tuned_{save_file}.pt")  # Save the fine-tuned model

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
    
    # Evaluate of the fine-tuned model on the validation set
    post_fine_tune_metrics = full_model_evaluation(
        model_after_fine_tuning,
        validation_loader,
        device
    )
    print("Post fine tune metrics")
    for item in post_fine_tune_metrics:
        print(f"{item}: ", post_fine_tune_metrics[item])