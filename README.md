# Data Mining Assignment Group 3

## Setup
Install uv: https://docs.astral.sh/uv/getting-started/installation/

    uv sync
    uv run python train_models.py

## AI-Generated Hotel Review Classification
This project investigates whether hotel reviews can be classified as human-written or AI-generated using machine learning. The dataset contains 'real' hotel reviews from Booking.com and AI-generated reviews created using GPT-4. The analysis is restricted to English-language reviews.

### Dataset
The original dataset is stored in: `all_data.csv`
The target variable is based on the source column: `0 = Human`, `1 = AI`

### Data Analysis
- Distribution of human and AI-generated reviews
- Word clouds for human and AI-generated reviews
- Review length analysis
- Frequency analysis of words

### Data Preprocessing
- Normalization
- Tokenization
- Stopword removal
- Stemming
- Duplicate removal

### Train/ Test-Split
The data is divided into **training/validation set (75%)**, and a **test set(25%)** using stratified random sampling.
Hyperparameter tuning is performed using cross-validation on the training data, so no separate hold-out validation set is required.
Run the `preprocessing.ipynb` to obtain the resulting datasets `train.csv` and `test.csv`.

### Feature Extraction

### ML Selection & Training

### ML Evaluation

### Project Structure
project/ 
│ 
├── all_data.csv 
├── train.csv 
├── test.csv 
├── preprocessing.ipynb 
├── README.md 
└── ...

### Requirements
The project uses Python and the following packages:

- `pandas`
- `nltk`
- `matplotlib`
- `wordcloud`
- `scikit-learn`

The NLTK stopword corpus is downloaded within: nltk.download("stopwords")
>>>>>>> origin/theresa
