import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency

# Configurations

RANDOM_STATE = 5
CROSS_VALIDATION_SPLITS = 5
TOP_N_FEATURES = 5
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
RESULTS_DIR = Path("results")
CLASS_NAMES = {0: "human", 1: "AI"}

logger = logging.getLogger(__name__)


# Helper functions


def setup_logging():
    """Log output in terminal and to a new file in RESULTS_DIR."""
    RESULTS_DIR.mkdir(exist_ok=True)
    log_file = RESULTS_DIR / f"run_{datetime.now():%Y-%m-%d_%H%M%S}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )


def print_metrics(name, y_true, y_pred):
    """Print accuracy, precision, recall, and F1 score for a model."""
    logger.info(f"\n--- {name} ---")
    logger.info(f"  Accuracy:   {accuracy_score(y_true, y_pred):.4f}")
    logger.info(f"  Precision:  {precision_score(y_true, y_pred, zero_division=0):.4f}")
    logger.info(f"  Recall:     {recall_score(y_true, y_pred, zero_division=0):.4f}")
    logger.info(f"  F1 Score:   {f1_score(y_true, y_pred, zero_division=0):.4f}")


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def mcnemar_test(y_true, pred1, pred2, name1, name2):
    """McNemar's test for comparing two classifiers."""
    correct1 = y_true == pred1
    correct2 = y_true == pred2

    # Create the 2x2 contingency table
    both_correct = np.sum(correct1 & correct2)
    only1_correct = np.sum(correct1 & ~correct2)
    only2_correct = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)

    disagreements = only1_correct + only2_correct
    stat = (
        (abs(only1_correct - only2_correct) - 1) ** 2 / disagreements
        if disagreements > 0
        else 0
    )
    p_value = 1 - chi2_dist.cdf(stat, df=1)

    logger.info(f"\n\n===== McNemar's Test: {name1} vs {name2} =====")
    logger.info(f"{'':30}{name2 + ' correct':>25}{name2 + ' incorrect':>25}")
    logger.info(f"{name1 + ' correct':30}{both_correct:>25}{only1_correct:>25}")
    logger.info(f"{name1 + ' incorrect':30}{only2_correct:>25}{both_wrong:>25}")
    logger.info(f"Stat: {stat:.4f}, p-value: {p_value:.4f}")

    return p_value


def get_feature_scores(model):
    """Get words and one score per word."""
    words = model.named_steps["vec"].get_feature_names_out()
    clf = model.named_steps["clf"]

    if isinstance(clf, MultinomialNB):
        words = model.named_steps["select"].get_feature_names_out(words)
        scores = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif isinstance(clf, LogisticRegression):
        scores = clf.coef_[0]
    else:
        scores = clf.feature_importances_
    return words, scores


def print_top_features(name, words, scores, signed):
    """Print the top features for a model.
    If signed is True, print the top positive and negative features separately."""
    if signed:
        order = np.argsort(scores)
        top_ai = order[-TOP_N_FEATURES:][::-1]  # Top positive
        top_human = order[:TOP_N_FEATURES]  # Top negative
        logger.info(f"\nTop {TOP_N_FEATURES} features for {name} (AI):")
        for idx in top_ai:
            logger.info(f"  {words[idx]}: {scores[idx]:.4f}")
        logger.info(f"\nTop {TOP_N_FEATURES} features for {name} (human):")
        for idx in top_human:
            logger.info(f"  {words[idx]}: {scores[idx]:.4f}")
    else:
        top_indices = np.argsort(scores)[::-1][:TOP_N_FEATURES]
        logger.info(f"\nTop {TOP_N_FEATURES} features for {name}:")
        for idx in top_indices:
            logger.info(f"  {words[idx]}: {scores[idx]:.4f}")


# Hyperparameter tuning


def tune(name, pipe, param_grid, X_train, y_train, cv):
    """Grid search with CV. Refit the best model on the full training set."""
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)

    logger.info(f"\n--- {name} ---")
    logger.info("  Best params:")
    for key, value in search.best_params_.items():
        formatted = f"{value:.4g}" if isinstance(value, float) else value
        # Drop the pipeline step prefix for printing (e.g. "clf__C" -> "C")
        logger.info(f"    {key.split('__')[-1]}: {formatted}")
    logger.info(f"  CV accuracy:  {search.best_score_:.4f}")
    return search.best_estimator_


def tune_logistic_regression(X_train, y_train, cv):
    """Tune Logistic Regression with L1 regularization."""
    pipe = Pipeline(
        [
            ("vec", CountVectorizer()),
            (
                "clf",
                LogisticRegression(
                    l1_ratio=1,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {"vec__min_df": [1, 2, 3, 5, 10], "clf__C": np.logspace(-2, 2, 15)}
    return tune("Logistic Regression", pipe, param_grid, X_train, y_train, cv)


def tune_naive_bayes(X_train, y_train, cv):
    """Tune Naive Bayes over min_df and k."""

    pipe = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("select", SelectKBest(chi2)),
            ("clf", MultinomialNB()),
        ]
    )

    param_grid = {
        "vec__min_df": [1, 2, 3, 5, 10],
        "select__k": [50, 100, 200, 300, 500, "all"],
    }
    return tune("Naive Bayes", pipe, param_grid, X_train, y_train, cv)


def tune_decision_tree(X_train, y_train, cv):
    """Tune Decision Tree with cost-complexity pruning.
    The best alpha is selected using cross-validation."""
    X_vec = CountVectorizer().fit_transform(X_train)
    ccp_path = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ).cost_complexity_pruning_path(X_vec, y_train)
    alphas = np.unique(ccp_path.ccp_alphas[:-1])

    pipe = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {"clf__ccp_alpha": alphas}
    return tune("Decision Tree", pipe, param_grid, X_train, y_train, cv)

def tune_random_forest(X_train, y_train, cv):
    """Tune Random Forest.
    The best min_df, max_features and n_estimators are selected using cross-validation."""
    pipe = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "vec__min_df": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2", 0.05, 0.1],
        "clf__n_estimators": [100, 300, 500],
    }
    return tune("Random Forest", pipe, param_grid, X_train, y_train, cv)

def tune_gradient_boosting(X_train, y_train, cv):
    """Tune Gradient Boosting.
    The best learning_rate, n_estimators and max_depth are selected using cross-validation."""
    pipe = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [1, 3, 5],
    }
    return tune("Gradient Boosting", pipe, param_grid, X_train, y_train, cv)

def accuracy_by_group_test(y_true, y_pred, groups, name):
    """Chi-square test of whether a model's accuracy differs between groups of reviews."""

    correct = np.where(np.asarray(y_true) == np.asarray(y_pred), "correct", "incorrect")
    table = pd.crosstab(np.asarray(groups), correct).reindex(
        columns=["correct", "incorrect"], fill_value=0
    )
    stat, p_value, _, _ = chi2_contingency(table)

    logger.info(f"\n\n===== Chi-square Test: {name} accuracy by review polarity =====")
    logger.info(f"{'':20}{'correct':>12}{'incorrect':>12}{'accuracy':>12}")
    for group, row in table.iterrows():
        accuracy = row["correct"] / row.sum()
        logger.info(
            f"{str(group):20}{row['correct']:>12}{row['incorrect']:>12}{accuracy:>12.4f}"
        )
    logger.info(f"Stat: {stat:.4f}, p-value: {p_value:.4f}")

    return p_value

def main():
    setup_logging()
    train_df, test_df = load_data()

    # Get the features and labels for training and testing
    X_train, y_train = train_df["review"], train_df["source"]
    X_test, y_test = test_df["review"], test_df["source"]

    logger.info("===== Settings =====")
    logger.info(f"  CV folds:     {CROSS_VALIDATION_SPLITS}")
    logger.info(f"  Random seed:  {RANDOM_STATE}")

    logger.info("\n\n===== Data =====")
    # Class counts with 0/1 shown as human/AI
    train_counts = y_train.value_counts().rename(CLASS_NAMES).to_dict()
    test_counts = y_test.value_counts().rename(CLASS_NAMES).to_dict()
    logger.info(f"  Train: {len(X_train)} reviews, {train_counts}")
    logger.info(f"  Test:  {len(X_test)} reviews, {test_counts}")

    logger.info("\n\n===== Hyperparameter Tuning (CV) =====")

    # Set up cross-validation splitter
    cv = StratifiedKFold(
        n_splits=CROSS_VALIDATION_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    tuned_models = {
    "Logistic Regression": tune_logistic_regression(X_train, y_train, cv),
    "Naive Bayes": tune_naive_bayes(X_train, y_train, cv),
    "Decision Tree": tune_decision_tree(X_train, y_train, cv),
    "Random Forest": tune_random_forest(X_train, y_train, cv),
    "Gradient Boosting": tune_gradient_boosting(X_train, y_train, cv),
    }

    # Evaluate the final models on the test set
    test_preds = {name: model.predict(X_test) for name, model in tuned_models.items()}

    logger.info("\n\n===== Test Set Results =====")
    for name, y_pred in test_preds.items():
        print_metrics(name, y_test, y_pred)

    # Get predictions comparing Naive Bayes and Logistic Regression for McNemar's test (Q1)
    mcnemar_test(
        y_test,
        test_preds["Logistic Regression"],
        test_preds["Naive Bayes"],
        "Logistic Regression",
        "Naive Bayes",
    )

    # McNemar's tests comparing each tree ensemble with the single tree (Q2)
    for name in ["Random Forest", "Gradient Boosting"]:
        mcnemar_test(
            y_test, test_preds[name], test_preds["Decision Tree"], name, "Decision Tree"
        )

    # Random Forest on positive vs negative reviews (Q3)
    logger.info("\n\n===== Random Forest by Review Polarity =====")
    polarity = test_df["Sentiment"]
    rf_pred = test_preds["Random Forest"]
    for value in sorted(polarity.unique()):
        mask = (polarity == value).to_numpy()
        print_metrics(f"Random Forest, {value} reviews", y_test[mask], rf_pred[mask])
    accuracy_by_group_test(y_test, rf_pred, polarity, "Random Forest")

    # Prints the top features for each model (Q4)
    logger.info("\n\n===== Top Features =====")
    for name, signed in [
        ("Naive Bayes", True), 
        ("Logistic Regression", True),
        ("Decision Tree", False),
        ("Random Forest", False),
        ("Gradient Boosting", False),
    ]:
        words, scores = get_feature_scores(tuned_models[name])
        print_top_features(name, words, scores, signed)


if __name__ == "__main__":
    main()
