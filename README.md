# BERT Fine-Tuning Project

This project focuses on fine-tuning a pre-trained BERT model for sequence classification tasks. It includes scripts for data preparation, model training, evaluation, and prediction. The project is structured to provide a clear workflow from raw data to a deployable machine learning model.

## Project Structure
```
.
├── README.md
├── constants.py - Constants used across the project.
├── main.py - The main script to run for training and evaluating the model.
├── models
│ ├── model.pt - Pre-trained or fine-tuned model weights.
| └── model_fine_tuned.pt - Fine-tuned model weights.
├── notebooks
│ └── model_fine_tuning.ipynb - Jupyter notebook for interactive model fine-tuning.
├── requirements.txt - Required libraries and dependencies.
├── scripts
│ ├── make_dataset.py - Script for data preparation and loading.
│ └── model.py - Script defining the model architecture and utilities.
├── setup.py - Setup script for installing the project as a package.
└── util.py - Utility functions used across the project.
```

## Setup

To set up the project environment, follow these steps:

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
````

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. (Optional) If you want to set up a virtual environment, run:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

To fine-tune and evaluate the BERT model, run the following command:

```bash
python main.py
```

For an interactive fine-tuning session, you can use the Jupyter notebook:

```bash
jupyter notebook notebooks/model_fine_tuning.ipynb
```

## Customization

- You can modify `constants.py` to change model training parameters such as epochs, batch size, learning rate, etc.
- To train on your own data, update the data loading mechanism in `scripts/make_dataset.py`.
- The model architecture can be adjusted in `scripts/model.py`.

## Evaluation

- Prior to fine tuning
    * Accuracy:     0.6485
    * F1-score:     0.5102
    * Recall:       0.6485
    * Precision:    0.4206
- Post fine tuning
    * Accuracy:     0.7629
    * F1-score:     0.7420
    * Recall:       0.7629
    * Precision:    0.7685

- Semantic Text Similarity Benchmark (STS test) evaluated against our model for Pearson correlation of sentences with range of 1-5 semantic similarity. Adjusted values to be rounded to 0 or 1 to match against our predictions for further examination.
    * Raw correlation calculated at 0.215
    * Adjusted correlation 0.182

