import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time


def load_data(path):
    """Load JSONL data and extract texts, labels, and original samples"""
    texts, labels, originals = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                texts.append(ex['raw_text'])
                labels.append(1 if ex['is_valid'] else 0)
                originals.append(ex)
    return texts, labels, originals


def run_logreg(train_file, val_file, test_file, output_file):
    """Train and evaluate logistic regression classifier on embeddings"""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    X_train_texts, y_train, _ = load_data(train_file)

    X_train = model.encode(X_train_texts, show_progress_bar=True)

    X_val_texts, y_val, _ = load_data(val_file)

    X_val = model.encode(X_val_texts, show_progress_bar=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_val_f1 = 0
    best_clf = None
    best_C = None

    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    for C in C_values:
        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train)

        val_preds = clf.predict(X_val_scaled)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='binary', zero_division=0
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_clf = clf
            best_C = C

    X_test_texts, y_test, test_examples = load_data(test_file)

    test_embed_start = time.time()
    X_test = model.encode(X_test_texts, show_progress_bar=True)
    test_embed_time = time.time() - test_embed_start

    X_test_scaled = scaler.transform(X_test)

    pred_start = time.time()
    preds = best_clf.predict(X_test_scaled)
    probs = best_clf.predict_proba(X_test_scaled)[:, 1]
    pred_time = time.time() - pred_start

    total_time_ms = ((test_embed_time + pred_time) / len(y_test)) * 1000

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='weighted', zero_division=0
    )
    accuracy = np.mean(preds == np.array(y_test))
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    print("TEST SET RESULTS")

    print("\nPerformance Metrics:")
    print(f"\tAccuracy:  {accuracy:.3f}")
    print(f"\tPrecision: {precision:.3f}")
    print(f"\tRecall:    {recall:.3f}")
    print(f"\tF1-Score:  {f1:.3f}")

    print("\nConfusion Matrix:")
    print(f"\tTP: {tp:>3}  FP: {fp:>3}")
    print(f"\tFN: {fn:>3}  TN: {tn:>3}")

    print("\nTiming:")
    print(f"\tEmbedding: {test_embed_time/len(y_test)*1000:.2f}ms/sample")
    print(f"\tPrediction: {pred_time/len(y_test)*1000:.2f}ms/sample")
    print(f"\tTotal: {total_time_ms:.2f}ms/sample")

    print(f"\nWriting predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for ex, pred, prob in zip(test_examples, preds, probs):
            ex['logreg_prediction'] = int(pred)
            ex['logreg_confidence'] = float(prob)
            ex['logreg_correct'] = (int(pred) == (1 if ex['is_valid'] else 0))
            f_out.write(json.dumps(ex) + "\n")

    print(f"Saved {len(test_examples)} predictions")

    stats_file = output_file.replace('.jsonl', '_stats.json')
    stats = {
        'model': 'LogisticRegression + all-MiniLM-L6-v2',
        'best_C': best_C,
        'embedding_dim': 384,
        'test_samples': len(y_test),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        },
        'avg_inference_time_ms': float(total_time_ms)
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("ERROR ANALYSIS")

    errors = [ex for ex, pred in zip(test_examples, preds)
              if pred != (1 if ex['is_valid'] else 0)]

    false_positives = [ex for ex, pred in zip(test_examples, preds)
                       if pred == 1 and not ex['is_valid']]
    false_negatives = [ex for ex, pred in zip(test_examples, preds)
                       if pred == 0 and ex['is_valid']]

    print(f"\nTotal errors: {len(errors)} ({len(errors)/len(y_test)*100:.1f}%)")
    print(f"\tFalse Positives: {len(false_positives)}")
    print(f"\tFalse Negatives: {len(false_negatives)}")

    if false_positives:
        print("\nFalse Positive Examples:")
        for i, ex in enumerate(false_positives[:3], 1):
            prob = probs[test_examples.index(ex)]
            print(f"\t{i}. {ex['filename']} (confidence: {prob:.3f})")

    if false_negatives:
        print("\nFalse Negative Examples:")
        for i, ex in enumerate(false_negatives[:3], 1):
            prob = probs[test_examples.index(ex)]
            print(f"\t{i}. {ex['filename']} (confidence: {prob:.3f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train LogReg classifier on embeddings")
    parser.add_argument("--train", default="train.jsonl", help="Training data")
    parser.add_argument("--val", default="val.jsonl", help="Validation data")
    parser.add_argument("--test", default="test.jsonl", help="Test data")
    parser.add_argument(
        "--output", default="logreg_results.jsonl", help="Output file")

    args = parser.parse_args()

    run_logreg(
        train_file=args.train,
        val_file=args.val,
        test_file=args.test,
        output_file=args.output
    )
