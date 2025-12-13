"""
Logistic classifier for PDF page relevance -> used before sending raw PDF pages to Claude via Bedrock
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).parent / "paper_texts_pdf_texts.csv"



class PageRelevanceClassifier:
    """Binary classifier for identifying relevant PDF pages for code gen"""

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.4):
        self.model_path = Path(model_path) if model_path else Path(__file__).parent / "page_classifier_model.pkl"
        self.threshold = threshold

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.trigram_vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.classifier: Optional[LogisticRegression] = None
        self.is_trained = False

        if self.model_path.exists():
            self.load_model()

    # Training
    def train(
        self,
        csv_path: str,
        test_size: float = 0.3,
        random_state: int = 42,
        oversample_positive: bool = True,
        unigram_max_features: int = 5000,
        trigram_max_features: int = 2000,
        c_regularization: float = 1.0,
        class_weight_ratio: float = 5.0,
        max_iter: int = 1000,
    ) -> Dict[str, Any]:

        df = self._load_labeled_dataframe(csv_path)
        X = df["text"].values
        y = df["keep"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=random_state
        )

        if oversample_positive:
            X_train, y_train = self._oversample_positive_pages(X_train, y_train, random_state)

        # TF-IDF 
        self.vectorizer = TfidfVectorizer(
            max_features=unigram_max_features,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        self.trigram_vectorizer = TfidfVectorizer(
            max_features=trigram_max_features,
            ngram_range=(3, 3),
            stop_words="english",
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
        )

        X_train_tf = self.vectorizer.fit_transform(X_train)
        X_train_tri = self.trigram_vectorizer.fit_transform(X_train)
        X_train_stats = self._extract_tfidf_statistics(X_train, X_train_tf)

        X_train_all = hstack([X_train_tf, X_train_tri, X_train_stats])

        X_test_tf = self.vectorizer.transform(X_test)
        X_test_tri = self.trigram_vectorizer.transform(X_test)
        X_test_stats = self._extract_tfidf_statistics(X_test, X_test_tf)

        X_test_all = hstack([X_test_tf, X_test_tri, X_test_stats])

        # Scaling features
        self.scaler = StandardScaler()
        n_stats = X_train_stats.shape[1]
        stat_start = X_train_all.shape[1] - n_stats

        X_train_scaled = self.scaler.fit_transform(X_train_all[:, stat_start:].toarray())
        X_test_scaled = self.scaler.transform(X_test_all[:, stat_start:].toarray())

        X_train_final = hstack([X_train_all[:, :stat_start], csr_matrix(X_train_scaled)])
        X_test_final = hstack([X_test_all[:, :stat_start], csr_matrix(X_test_scaled)])

        # Model Definition (logistic regression)
        self.classifier = LogisticRegression(
            C=c_regularization,
            max_iter=max_iter,
            class_weight={0: 1.0, 1: class_weight_ratio},
            solver="lbfgs",
            random_state=random_state,
        )

        self.classifier.fit(X_train_final, y_train)
        self.is_trained = True

        # Evaluation
        probs = self.classifier.predict_proba(X_test_final)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        logger.info("Accuracy: %.4f", acc)
        logger.info("\n%s", classification_report(y_test, preds, target_names=["Not Relevant", "Relevant"]))

        # Threshold sweep
        best_f1, best_thresh = 0.0, self.threshold
        for t in np.arange(0.3, 0.8, 0.05):
            f1 = f1_score(y_test, (probs >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        logger.info("Best threshold=%.2f (F1=%.3f)", best_thresh, best_f1)
        self.threshold = best_thresh

        self.save_model()

        return {
            "accuracy": acc,
            "threshold": self.threshold,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }


    def evaluate_on_csv(self, csv_path: str) -> Dict[str, float]:
        df = self._load_labeled_dataframe(csv_path)
        X = df["text"].values
        y = df["keep"].values

        preds = self.predict_batch(X)
        y_pred = np.array([int(p["is_relevant"]) for p in preds])

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }


    def predict(self, page_text: str) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        tf = self.vectorizer.transform([page_text])
        tri = self.trigram_vectorizer.transform([page_text])
        stats = self._extract_tfidf_statistics([page_text], tf)

        combined = hstack([tf, tri, stats])
        stat_start = combined.shape[1] - stats.shape[1]

        stats_scaled = self.scaler.transform(combined[:, stat_start:].toarray())
        final = hstack([combined[:, :stat_start], csr_matrix(stats_scaled)])

        prob = self.classifier.predict_proba(final)[0, 1]
        return {
            "is_relevant": prob >= self.threshold,
            "probability": float(prob),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]


    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.__dict__.update(pickle.load(f))
        logger.info("Loaded model from %s", self.model_path)


    # Helpers
    def _load_labeled_dataframe(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df[df["keep"].isin([0, 1])]
        df["text"] = df["text"].fillna("").str.replace("\n", " ")
        return df

    def _oversample_positive_pages(
        self, texts: np.ndarray, labels: np.ndarray, random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]

        if len(pos) == 0 or len(neg) == 0:
            return texts, labels

        rng = np.random.default_rng(random_state)
        pos_sample = rng.choice(pos, size=len(neg), replace=True)
        idx = np.concatenate([neg, pos_sample])
        rng.shuffle(idx)

        return texts[idx], labels[idx]

    def _extract_tfidf_statistics(self, texts: List[str], tfidf_matrix) -> csr_matrix:
        rows = []
        for i, text in enumerate(texts):
            v = tfidf_matrix[i].toarray().ravel()
            words = text.split()

            rows.append([
                v.sum(),
                v.mean(),
                v.std(),
                v.max(),
                np.count_nonzero(v),
                len(words),
                len(text),
                text.count("="),
                text.count("("),
                len([w for w in words if len(w) > 10]),
            ])
        return csr_matrix(rows)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv")
    parser.add_argument("--evaluate-csv")
    parser.add_argument("--model-path")
    args = parser.parse_args()

    clf = PageRelevanceClassifier(model_path=args.model_path)

    if args.train_csv:
        clf.train(args.train_csv)

    if args.evaluate_csv:
        metrics = clf.evaluate_on_csv(args.evaluate_csv)
        logger.info("Evaluation: %s", metrics)


if __name__ == "__main__":
    main()