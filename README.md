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
The file `analysis.ipynb` is intended for data understanding and exploratory analysis and is not required to reproduce the final results. It includes the following:
- Distribution of human and AI-generated reviews
- Word clouds for human and AI-generated reviews
- Review length analysis
- Frequency analysis of words
- Analysis of contractions, punctuation, and numerical content 
Overall, these analyses provide a rationale for the subsequent preprocessing decisions.

### Data Preprocessing
All preprocessing steps are summarized in `preprocessing.ipynb`. This includes:
- Normalization
- Tokenization
- Stopword removal
- Stemming

### Train/ Test-Split
The data is divided into **training/validation set (75%)**, and a **test set(25%)** using stratified random sampling.
Hyperparameter tuning is performed using cross-validation on the training data, so no separate hold-out validation set is required.
Run the file `preprocessing.ipynb` to obtain the training (`train_v.1.csv`) and test file (`test_v.1.csv`).

### Feature Extraction

### ML Selection & Training

### ML Evaluation


<!-- ### Project Structure
Do we need this section? (Comment: Theresa)
project/ 
│ 
├── all_data.csv 
├── train.csv 
├── test.csv 
├── preprocessing.ipynb 
├── README.md 
└── ... -->

### Requirements
The project uses Python and the following external packages:

- `pandas`
- `nltk`
- `matplotlib`
- `wordcloud`
- `scikit-learn`

The following Python standard library modules are also used and do not require separate installation:
- `re`
- `string`
- `unicodedata`
- `collections (for Counter)`

Additionally, the required NLTK resources, such as English stopwords, must be downloaded before running the notebook.
