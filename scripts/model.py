import torch
import torch.nn as nn
from transformers import BertModel

from constants import TOKENIZER_MAX_LENGTH

class BertForSequenceClassificationCustom(nn.Module):
    """
    A custom class for sequence classification that builds upon the pre-trained BERT model.
    
    Attributes:
        num_labels (int): The number of labels for the classification task.
        bert (BertModel): Pre-trained BERT model from Hugging Face Transformers.
        dropout (nn.Dropout): Dropout layer to reduce overfitting.
        classifier (nn.Sequential): Custom classifier layers added on top of BERT.
    """
    def __init__(self, num_labels=2):
        """
        Initializes the model by setting up the layers.
        
        Parameters:
            num_labels (int): Number of target labels for classification.
        """
        super(BertForSequenceClassificationCustom, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)  # Dropout layer for regularizing the model
        
        # Custom classifier that is added on top of the BERT model
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),  # 768 is the dimensionality of BERT's output
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Another dropout layer for the classifier
            nn.Linear(512, num_labels)  # Final layer for classification
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.

        Parameters:
            input_ids (torch.Tensor): Indices of input sequence tokens.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding tokens.
            token_type_ids (torch.Tensor): Segment token indices.

        Returns:
            torch.Tensor: Logits from the classifier.
        """
        # Obtain the encoded sequence from BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # We are interested in BERT's pooled output, typically used for classification tasks
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        logits = self.classifier(pooled_output)  # Obtain logits from the classifier
        
        return logits


def predict_on_example(model, tokenizer, sentence1, sentence2, device):
    """
    Make a prediction for a pair of sentences.

    Parameters:
        model (BertForSequenceClassificationCustom): The trained model for prediction.
        tokenizer (Tokenizer): Tokenizer for processing the text.
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        device (torch.device): The device to perform computation on.

    Returns:
        Tuple: Probabilities and predicted label.
    """
    model.eval()  # Set the model to evaluation mode
    # Tokenize the input sentences
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )
    # Move tensors to the specified device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():  # Disable gradient computation
        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    probabilities = torch.softmax(logits, dim=1)  # Calculate probabilities
    prediction = torch.argmax(probabilities, dim=1)  # Get the predicted label
    
    return probabilities, prediction.item()


def get_base_model():
    """
    Initializes and returns the custom BERT model.

    Returns:
        BertForSequenceClassificationCustom: The initialized model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the computation device
    
    # Initialize the custom model
    model = BertForSequenceClassificationCustom(num_labels=2)
    model.to(device)  # Move the model to the device
    
    return model
