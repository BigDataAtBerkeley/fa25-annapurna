"""
Logistic classifer for page relevance (for when we send a PDF to the bedrock vision api)
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_DATA_PATH = Path(__file__).parent / "paper_texts - pdf_texts.csv"



class PageRelevanceClassifier:
    """Classifier to determine if a PDF page is relevant for code generation."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the page relevance classifier.
        
        Args:
            model_path: Path to saved model file. If None, will use default path.
        """
        # Default model path in code_gen directory
        if model_path is None:
            model_path = Path(__file__).parent / "page_classifier_model.pkl"
        
        self.model_path = Path(model_path)
        self.vectorizer = None
        self.trigram_vectorizer = None
        self.scaler = None
        self.classifier = None
        self.is_trained = False
        
        # Try to load existing model
        if self.model_path.exists():
            try:
                self.load_model()
                logger.info(f"Loaded page classifier from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}")
                logger.info("Will train a new model")
    
    def train(
        self,
        csv_path: str,
        test_size: float = 0.3,
        random_state: int = 42,
        oversample_positive: bool = True,
        unigram_max_features: int = 5000,
        trigram_max_features: int = 2000
    ):
        """
        Train the classifier on labeled page data.
        
        Args:
            csv_path: Path to CSV file with columns: paper_id, pdf_name, page_number, text, keep
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            oversample_positive: Whether to balance training data by oversampling
                the minority (keep=1) class after the train/test split. Enabled by default.
            unigram_max_features: Vocabulary size for the main TF-IDF vectorizer.
            trigram_max_features: Vocabulary size for the trigram TF-IDF vectorizer.
        """
        logger.info(f"Training page classifier on data from {csv_path}")
        
        # Load and clean data
        df = self._load_labeled_dataframe(csv_path)
        
        logger.info(f"Training on {len(df)} pages")
        logger.info(f"Class distribution: {df['keep'].value_counts().to_dict()}")
        
        # Prepare features and labels
        X = df['text'].fillna('').astype(str).values
        y = df['keep'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if oversample_positive:
            X_train, y_train = self._oversample_positive_pages(
                X_train, y_train, random_state
            )
            logger.info(
                "Applied positive-class oversampling. New train distribution: %s",
                {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
            )
        
        # Create TF-IDF vectorizers with different configurations for richer features
        # Main TF-IDF vectorizer with unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            max_features=unigram_max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english',
            sublinear_tf=True,  # Use logarithmic scaling for term frequency
            norm='l2'  # L2 normalization
        )
        
        # Additional TF-IDF vectorizer for trigrams (captures longer phrases)
        self.trigram_vectorizer = TfidfVectorizer(
            max_features=trigram_max_features,
            ngram_range=(3, 3),  # Trigrams only
            min_df=3,
            max_df=0.9,
            stop_words='english',
            sublinear_tf=True,
            norm='l2'
        )
        
        # Fit vectorizers and transform training data
        logger.info("Extracting TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_train_trigram = self.trigram_vectorizer.fit_transform(X_train)
        
        # Extract additional TF-IDF-based statistical features
        X_train_stats = self._extract_tfidf_statistics(X_train, X_train_tfidf)
        
        # Combine all features
        X_train_combined = hstack([X_train_tfidf, X_train_trigram, X_train_stats])
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        X_test_trigram = self.trigram_vectorizer.transform(X_test)
        X_test_stats = self._extract_tfidf_statistics(X_test, X_test_tfidf)
        X_test_combined = hstack([X_test_tfidf, X_test_trigram, X_test_stats])
        
        # Normalize statistical features
        self.scaler = StandardScaler()
        # Get indices for statistical features (last N features)
        n_stats = X_train_stats.shape[1]
        stats_start_idx = X_train_combined.shape[1] - n_stats
        
        # Extract and scale statistical features
        X_train_stats_dense = X_train_combined[:, stats_start_idx:].toarray()
        X_test_stats_dense = X_test_combined[:, stats_start_idx:].toarray()
        
        X_train_stats_scaled = self.scaler.fit_transform(X_train_stats_dense)
        X_test_stats_scaled = self.scaler.transform(X_test_stats_dense)
        
        # Recombine with scaled stats
        X_train_final = hstack([
            X_train_combined[:, :stats_start_idx],
            csr_matrix(X_train_stats_scaled)
        ])
        X_test_final = hstack([
            X_test_combined[:, :stats_start_idx],
            csr_matrix(X_test_stats_scaled)
        ])
        
        # Train logistic regression classifier
        self.classifier = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',  # Good for small-medium datasets
            C=1.0  # Regularization strength
        )
        
        logger.info("Training classifier...")
        self.classifier.fit(X_train_final, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Not Relevant', 'Relevant']))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'model_path': str(self.model_path)
        }

        
    def predict(self, page_text: str) -> Dict[str, Any]:
        """
        Predict if a page is relevant for code generation.
        
        Args:
            page_text: Text content of the page
            
        Returns:
            Dictionary with prediction and confidence score
        """
        if not self.is_trained or self.vectorizer is None or self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first or load a saved model.")
        
        # Handle empty text
        if not page_text or not page_text.strip():
            return {
                'is_relevant': False,
                'confidence': 0.0,
                'probability': 0.0
            }
        
        # Transform text with all feature extractors
        text_tfidf = self.vectorizer.transform([str(page_text)])
        text_trigram = self.trigram_vectorizer.transform([str(page_text)])
        text_stats = self._extract_tfidf_statistics([str(page_text)], text_tfidf)
        
        # Combine features
        text_combined = hstack([text_tfidf, text_trigram, text_stats])
        
        # Scale statistical features
        n_stats = text_stats.shape[1]
        stats_start_idx = text_combined.shape[1] - n_stats
        text_stats_dense = text_combined[:, stats_start_idx:].toarray()
        text_stats_scaled = self.scaler.transform(text_stats_dense)
        
        # Recombine with scaled stats
        text_final = hstack([
            text_combined[:, :stats_start_idx],
            csr_matrix(text_stats_scaled)
        ])
        
        # Predict
        prediction = self.classifier.predict(text_final)[0]
        probabilities = self.classifier.predict_proba(text_final)[0]
        
        # Get probability of relevant class (class 1)
        relevant_prob = probabilities[1] if len(probabilities) > 1 else 0.0
        
        return {
            'is_relevant': bool(prediction),
            'confidence': float(relevant_prob),
            'probability': float(relevant_prob)
        }
    
    def predict_batch(self, page_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict relevance for multiple pages.
        
        Args:
            page_texts: List of page text contents
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained or self.vectorizer is None or self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first or load a saved model.")
        
        # Handle empty texts
        page_texts = [str(text) if text else '' for text in page_texts]
        
        # Transform texts with all feature extractors
        text_tfidf = self.vectorizer.transform(page_texts)
        text_trigram = self.trigram_vectorizer.transform(page_texts)
        text_stats = self._extract_tfidf_statistics(page_texts, text_tfidf)
        
        # Combine features
        text_combined = hstack([text_tfidf, text_trigram, text_stats])
        
        # Scale statistical features
        n_stats = text_stats.shape[1]
        stats_start_idx = text_combined.shape[1] - n_stats
        text_stats_dense = text_combined[:, stats_start_idx:].toarray()
        text_stats_scaled = self.scaler.transform(text_stats_dense)
        
        # Recombine with scaled stats
        text_final = hstack([
            text_combined[:, :stats_start_idx],
            csr_matrix(text_stats_scaled)
        ])
        
        # Predict
        predictions = self.classifier.predict(text_final)
        probabilities = self.classifier.predict_proba(text_final)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            relevant_prob = probs[1] if len(probs) > 1 else 0.0
            results.append({
                'is_relevant': bool(pred),
                'confidence': float(relevant_prob),
                'probability': float(relevant_prob)
            })
        
        return results
    
    def evaluate_on_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Evaluate the trained model on a labeled CSV.
        
        Args:
            csv_path: Path to CSV with `text` and `keep` columns.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained or loaded before evaluating.")
        
        df = self._load_labeled_dataframe(csv_path)
        
        texts = df['text'].fillna('').astype(str).tolist()
        labels = df['keep'].tolist()
        
        logger.info(f"Evaluating on {len(df)} pages from {csv_path}")
        preds = self.predict_batch(texts)
        pred_labels = [int(p['is_relevant']) for p in preds]
        
        accuracy = accuracy_score(labels, pred_labels)
        report = classification_report(
            labels,
            pred_labels,
            target_names=['Not Relevant', 'Relevant']
        )
        
        logger.info(f"Evaluation accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(report)
        
        return {
            'accuracy': accuracy,
            'sample_size': len(df),
            'report': report
        }
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'trigram_vectorizer': self.trigram_vectorizer,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {save_path}")
    
    def load_model(self, path: Optional[str] = None):
        """Load a trained model from disk."""
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.trigram_vectorizer = model_data.get('trigram_vectorizer')
        self.scaler = model_data.get('scaler')
        self.classifier = model_data['classifier']
        self.is_trained = model_data.get('is_trained', True)
        
        # Handle backward compatibility (old models without trigram vectorizer)
        if self.trigram_vectorizer is None:
            logger.warning("Loaded model doesn't have trigram vectorizer. Retraining recommended.")
            # Create a dummy trigram vectorizer that returns empty features
            # Use a minimal vocabulary to avoid the max_features=0 error
            self.trigram_vectorizer = TfidfVectorizer(
                max_features=1,
                ngram_range=(3, 3),
                vocabulary=['dummy_token_never_appears']
            )
            self.trigram_vectorizer.fit(['dummy'])  # Fit on dummy to initialize
        
        if self.scaler is None:
            logger.warning("Loaded model doesn't have scaler. Retraining recommended.")
            self.scaler = StandardScaler()
            # Fit on dummy data to initialize
            self.scaler.fit([[0] * 16])  # 16 is the number of statistical features
        
        logger.info(f"Loaded model from {load_path}")
    
    def _load_labeled_dataframe(self, csv_path: str) -> pd.DataFrame:
        """Load CSV, ensure required columns, and clean keep labels."""
        df = pd.read_csv(csv_path)
        required_cols = ['text', 'keep']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must have columns: {required_cols}")
        
        df = df[df['text'].notna() & df['keep'].notna()].copy()
        
        # Normalize keep column to contain only digits 0/1
        keep_series = (
            df['keep']
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace('o', '0')
            .str.replace(r'[^01]', '', regex=True)
        )
        keep_series = keep_series.str.extract(r'([01])')[0]
        df['keep'] = keep_series
        df = df[df['keep'].notna()].copy()
        df['keep'] = df['keep'].astype(int)
        df = df[df['keep'].isin([0, 1])]
        
        if len(df) == 0:
            raise ValueError("No valid labeled data found")
        
        return df
    
    def _oversample_positive_pages(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample positive class to match negatives in the training split."""
        positives = np.where(labels == 1)[0]
        negatives = np.where(labels == 0)[0]
        
        if len(positives) == 0 or len(negatives) == 0:
            logger.warning("Skipping oversampling due to missing class in training split.")
            return texts, labels
        
        if len(positives) >= len(negatives):
            return texts, labels
        
        rng = np.random.default_rng(seed=random_state)
        sampled_pos = rng.choice(
            positives,
            size=len(negatives),
            replace=True
        )
        new_indices = np.concatenate([negatives, sampled_pos])
        rng.shuffle(new_indices)
        
        return texts[new_indices], labels[new_indices]
    
    def _extract_tfidf_statistics(self, texts: List[str], tfidf_matrix) -> csr_matrix:
        """
        Extract TF-IDF-based statistical features from text.
        
        Args:
            texts: List of text strings
            tfidf_matrix: TF-IDF matrix for the texts
            
        Returns:
            Sparse matrix with statistical features
        """
        features = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            text_words = text_lower.split()
            
            # Get TF-IDF vector for this document
            doc_tfidf = tfidf_matrix[i].toarray().flatten()
            
            # Statistical features from TF-IDF
            stats = [
                np.sum(doc_tfidf),  # Sum of all TF-IDF scores
                np.mean(doc_tfidf),  # Mean TF-IDF score
                np.std(doc_tfidf),  # Standard deviation of TF-IDF scores
                np.max(doc_tfidf),  # Maximum TF-IDF score
                np.percentile(doc_tfidf, 75),  # 75th percentile
                np.percentile(doc_tfidf, 50),  # Median
                np.percentile(doc_tfidf, 25),  # 25th percentile
                np.sum(doc_tfidf > 0),  # Number of non-zero TF-IDF features
                len(text_words),  # Document length (word count)
                len(text) / max(len(text_words), 1),  # Average word length
            ]
            
            # Additional text-based features that complement TF-IDF
            # These help capture structural information
            stats.extend([
                text.count('\n'),  # Number of lines
                text.count('.'),  # Number of sentences (approximate)
                text.count('='),  # Equations
                text.count('('),  # Parentheses (often in formulas)
                text.count('['),  # Brackets (often in formulas)
                len([w for w in text_words if len(w) > 10]),  # Long words (technical terms)
            ])
            
            features.append(stats)
        
        return csr_matrix(features)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the PageRelevanceClassifier."
    )
    default_train = str(DEFAULT_DATA_PATH) if DEFAULT_DATA_PATH.exists() else None
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to save/load the trained model (defaults to code_gen/page_classifier_model.pkl)."
    )
    parser.add_argument(
        "--train-csv",
        default=default_train,
        help=(
            "CSV of labeled pages used for training "
            f"(defaults to {DEFAULT_DATA_PATH.name} if it exists)."
        )
    )
    parser.add_argument(
        "--evaluate-csv",
        help="CSV of labeled pages to evaluate after training or loading a model."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of the training CSV to reserve for validation (default: 0.3)."
    )
    parser.add_argument(
        "--no-oversample-positive",
        action="store_false",
        dest="oversample_positive",
        help="Disable balancing the training set by oversampling pages with keep=1."
    )
    parser.set_defaults(oversample_positive=True)
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splitting and oversampling."
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=5000,
        help="Max vocabulary size for the main (1-2 gram) TF-IDF vectorizer."
    )
    parser.add_argument(
        "--trigram-max-features",
        type=int,
        default=2000,
        help="Max vocabulary size for the trigram TF-IDF vectorizer."
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    
    if not args.train_csv and not args.evaluate_csv:
        raise SystemExit("Specify --train-csv, --evaluate-csv, or both.")
    
    classifier = PageRelevanceClassifier(model_path=args.model_path)
    
    if args.train_csv:
        train_path = Path(args.train_csv)
        if not train_path.exists():
            raise SystemExit(f"Training CSV not found: {train_path}")
        train_metrics = classifier.train(
            args.train_csv,
            test_size=args.test_size,
            random_state=args.random_state,
            oversample_positive=args.oversample_positive,
            unigram_max_features=args.tfidf_max_features,
            trigram_max_features=args.trigram_max_features
        )
        logger.info(f"Training metrics: {train_metrics}")
    
    if args.evaluate_csv:
        eval_path = Path(args.evaluate_csv)
        if not eval_path.exists():
            raise SystemExit(f"Evaluation CSV not found: {eval_path}")
        eval_metrics = classifier.evaluate_on_csv(args.evaluate_csv)
        logger.info(f"Evaluation metrics: {eval_metrics}")


if __name__ == "__main__":
    main()