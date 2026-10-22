import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import unicodedata

nltk.download("stopwords")

df = pd.read_csv("data/all_data.csv")

# restrict analysis to the English language reviews. 
df = df[df["Review_Language"] == "English"].copy()

df["Review"] = (df["Upside_Review"].fillna("") + " " +df["Downside_Review"].fillna(""))
df["Review"] = df["Review"].str.strip()

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def normalize_unicode(text):
    # NFC: canonical decomposition followed by canonical composition
    return unicodedata.normalize("NFC", text)

def remove_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def preprocess_text(text):
    # 1. Normalization
    text = normalize_unicode(text)
    text = remove_accents(text)
    text = text.lower()

    # Keep numbers and punctuation
    # Do not expand contractions
    # Remove unwanted characters only
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"()\-]", " ", text)

    # Remove multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # 2. Tokenization
    tokens = text.split()

    # 3. Stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # 4. Porter stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

df["Review_clean"] = df["Review"].apply(preprocess_text)

df_model = df[["Review_clean", "source", "Sentiment"]].copy()

# check for duplicates
print("duplicate reviews:", df_model["Review_clean"].duplicated().sum())

df_model = df_model.rename(columns={"Review_clean": "review"})

#  Split the data randomly into a training/validation set (75%) and a test set(25%).
#  Use stratified random sampling so that the class distribution in the training set and test set are the same.

X = df_model["review"]
y = df_model["source"]

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, df_model["Sentiment"], test_size=0.25, stratify=y, random_state=42
)

train_df = pd.DataFrame({"review": X_train, "source": y_train, "Sentiment": s_train})
test_df = pd.DataFrame({"review": X_test, "source": y_test, "Sentiment": s_test})
train_df.to_csv("data/train_v.1.csv", index=False)
test_df.to_csv("data/test_v.1.csv", index=False)
