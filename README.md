# BERT Fine-Tuning Project

This project focuses on fine-tuning a pre-trained BERT model for sequence classification tasks. It includes scripts for data preparation, model training, evaluation, and prediction. The project is structured to provide a clear workflow from raw data to a deployable machine learning model.

## Project Structure
```
.
├── README.md
├── data
│   ├── processed
│   │   └── medicalCorpus.csv
│   └── raw
│       └── medicalCorpus.txt
├── notebooks
│   ├── data_file.ipynb
│   ├── evaluation.ipynb
│   └── mode_fine_tuning.ipynb
├── requirements.txt
└── scripts
    ├── constants.py
    ├── main.py
    ├── make_dataset.py
    ├── model.py
    ├── setup.py
    └── util.py
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
### GLUE-MRPC
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

### Medical Domain Dataset
- Base Model
    * F1-score: 0.599
- Medical Fine-tuned:
    * F1-score: 0.601
- GLUE-MRPC Fine-tuned:
    * F1-score: 0.655
- Multiple Fine-tuned:
    * F1-score: 0.750
