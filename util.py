import torch
import numpy as np

def save_model(model, name):
    """
    Save the PyTorch model state dictionary.

    Parameters:
    model (torch.nn.Module): The model to be saved.
    name (str): The name of the file to save the model state dictionary.
    """
    # Save the model's state dictionary under the 'models/' directory
    torch.save(model.state_dict(), f"models/{name}")
    print("Model saved.")  # Confirmation message


def flat_accuracy(preds, labels):
    """
    Calculate the accuracy of predictions compared to labels.

    Parameters:
    preds (numpy.ndarray): Predictions array with shape (num_samples, num_classes).
    labels (numpy.ndarray): Ground truth labels array with shape (num_samples,).

    Returns:
    float: The accuracy as a proportion of correct predictions.
    """
    # Flatten predictions and labels for comparison
    pred_flat = np.argmax(preds, axis=1).flatten()  # Convert softmax predictions to class predictions
    labels_flat = labels.flatten()  # Flatten the labels
    return np.sum(pred_flat == labels_flat) / len(labels_flat)  # Calculate and return accuracy


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model's performance on a dataset.

    Parameters:
    model (torch.nn.Module): The model to be evaluated.
    dataloader (torch.utils.data.DataLoader): The DataLoader containing the evaluation dataset.
    device (torch.device): The device to perform the evaluation on.

    Returns:
    float: The average accuracy across all batches in the dataloader.
    """
    model.eval()  # Set the model to evaluation mode
    total_eval_accuracy = 0  # Accumulator for the total accuracy

    # Iterate over all batches in the provided DataLoader
    for batch in dataloader:
        # Move batch data to the device
        batch = tuple(t.to(device) for t in batch)
        # Unpack the batch data
        b_input_ids, b_input_mask, b_labels = batch

        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Get model predictions for the current batch
            logits = model(b_input_ids, attention_mask=b_input_mask)

        # Move logits and labels to CPU for further operations
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Accumulate the total accuracy
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Compute the average accuracy across all batches
    return total_eval_accuracy / len(dataloader)
