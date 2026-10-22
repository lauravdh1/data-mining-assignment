import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Data Input
df = pd.read_csv("data/all_data.csv")
# Restrict analysis to the English language reviews
df = df[df["Review_Language"] == "English"].copy()
df["target"] = df["source"].map({0: "Human",1: "AI"})


df["Review"] = (df["Upside_Review"].fillna("") + " " +df["Downside_Review"].fillna(""))
df["Review"] = df["Review"].str.strip()

# Data distribution
source_counts = df["source"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
source_counts.plot(kind="bar",color=["blue", "orange"])
plt.xlabel("Review type")
plt.ylabel("Number of reviews")
plt.xticks([0, 1], ["Human", "AI"], rotation=0)
plt.show()

# Combine review columns
df["Review"] = (df["Upside_Review"].fillna("") + " " +df["Downside_Review"].fillna(""))
df["Review"] = df["Review"].str.strip()

# Word Clouds: visualizing a review's most significant and frequently occurring words.
# Word Cloud Human
human_text = " ".join(df.loc[df["source"] == 0, "Review"].dropna())
wordcloud_human = WordCloud(width=1000, height=500,background_color="white").generate(human_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_human, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Human")
plt.show()

# Word Cloud Fake
fake_text = " ".join(df.loc[df["source"] == 1, "Review"].dropna())
wordcloud_fake = WordCloud(width=1000, height=500, background_color="white").generate(fake_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_fake, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - AI")
plt.show()

# Evaluate differnces in vocabulary / word frequencies
def get_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

human_words = get_words(" ".join(df.loc[df["source"] == 0, "Review"].dropna()))
fake_words = get_words(" ".join(df.loc[df["source"] == 1, "Review"].dropna()))
human_common = Counter(human_words).most_common(20)
fake_common = Counter(fake_words).most_common(20)
print("Human:", human_common)
print("\nAI:", fake_common)

# Evaluate differences in overall word count
df["word_count"] = df["Review"].fillna("").str.split().str.len()
print(df.groupby("source")["word_count"].agg(["count", "mean", "median", "std", "min", "max"]))

df_contr = df.copy()

# Normalize apostrophe types to ease detection 
df_contr["Review_analysis"] = (df_contr["Review"].fillna("").str.replace("’", "'", regex=False))

# Common English contractions
contraction_pattern = re.compile(
    r"\b(?:"
    r"[a-z]+n't|"                       # don't, didn't, isn't
    r"[a-z]+'(?:m|re|ve|ll|d)\b|"       # I'm, we're, they've, I'll
    r"it's|that's|there's|here's|"
    r"what's|who's|let's|"
    r"y'all|gonna|wanna|gotta"
    r")\b",
    flags=re.IGNORECASE
)

# Count contractions and words per review
df_contr["contraction_count"] = (df_contr["Review_analysis"].apply(lambda x: len(contraction_pattern.findall(x))))

df_contr["word_count"] = (df_contr["Review_analysis"].str.findall(r"\b[a-zA-Z]+\b").str.len())

df_contr["has_contraction"] = (df_contr["contraction_count"] > 0)

summary = df_contr.groupby("source").agg(
    reviews=("Review", "size"),
    reviews_with_contractions=("has_contraction", "sum"),
    total_contractions=("contraction_count", "sum"),
    total_words=("word_count", "sum")
)

summary["percentage_with_contractions"] = (
    summary["reviews_with_contractions"]
    / summary["reviews"] * 100
)

summary["contractions_per_1000_words"] = (
    summary["total_contractions"]
    / summary["total_words"] * 1000
)

print(summary.round(2))

df_analysis = df.copy()
df_analysis["Review"] = df_analysis["Review"].fillna("")

# Word count
df_analysis["word_count"] = (df_analysis["Review"].str.findall(r"\b\w+\b").str.len())

# Count numeric sequences, e.g. 22, 2025, 4.5
df_analysis["number_count"] = (df_analysis["Review"].str.findall(r"\d+(?:[.,]\d+)?").str.len())

# Count selected punctuation marks
for char, name in [
    ("?", "question_marks"),
    ("!", "exclamation_marks"),
    (",", "commas"),
    (".", "periods"),
    (";", "semicolons"),
    (":", "colons"),
    ("'", "apostrophes"),
    ('"', "quotation_marks"),
]:
    df_analysis[name] = df_analysis["Review"].str.count(
        re.escape(char)
    )

# Summarize by source
count_columns = [
    "number_count",
    "question_marks",
    "exclamation_marks",
    "commas",
    "periods",
    "semicolons",
    "colons",
    "apostrophes",
    "quotation_marks",
]

summary = df_analysis.groupby("source").agg(
    reviews=("Review", "size"),
    total_words=("word_count", "sum"),
    **{
        col: (col, "sum")
        for col in count_columns
    }
)

# Normalize frequencies by text length
for col in count_columns:
    summary[f"{col}_per_1000_words"] = (
        summary[col] / summary["total_words"] * 1000
    )

print(summary.T)
