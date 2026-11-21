# Remove warning text from torchtext version
import warnings
warnings.filterwarnings("ignore")
import torchtext
torchtext.disable_torchtext_deprecation_warning()

import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from torchtext.datasets import AG_NEWS
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

SEED = 42
MAX_FEATURES = 20000
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# 1. Load, inspect, and vectorize the data

# 1a. Load training and testing examples from torchtext
train_data = list(AG_NEWS(split="train"))
test_data = list(AG_NEWS(split="test"))

# 1b. Combine each article and description into one text string
train_label = np.array([label - 1 for label, text in train_data])
train_text = [text for label, text in train_data]
test_label = np.array([label - 1 for label, text in test_data])
test_text = [text for label, text in test_data]

# 1c. Display one example article from each category & number of examples per split
print(f"Train size: {len(train_text)}, Test size: {len(test_text)}")
print("\nOne example per category:")
for c, name in enumerate(CLASS_NAMES):
    i = int(np.where(train_label == c)[0][0])
    print(f"\n[{name}]\n{train_text[i]}")

train_counts, test_counts = Counter(train_label.tolist()), Counter(test_label.tolist())
print(f"\n{'Category':<10} {'Train':>8} {'Test':>8}")
for c, name in enumerate(CLASS_NAMES):
    print(f"{name:<10} {train_counts[c]:>8} {test_counts[c]:>8}")

# 1d. Create a document term matrix (DTM) & TF-IDF feature matrix
def make_vectorizer(kind, remove_stop):
    cls = CountVectorizer if kind == "Count" else TfidfVectorizer
    return cls(max_features=MAX_FEATURES, stop_words="english" if remove_stop else None)

count_vec = make_vectorizer("Count", False)
X_train_count = count_vec.fit_transform(train_text)
X_test_count = count_vec.transform(test_text)

tfidf_vec = make_vectorizer("Tfidf", False)
X_train_tfidf = tfidf_vec.fit_transform(train_text)
X_test_tfidf = tfidf_vec.transform(test_text)

# 1e. Report shape of training DTM
rows, cols = X_train_count.shape
sparsity = (1 - X_train_count.nnz / (rows * cols)) * 100
print(f"\nTraining DTM shape: {rows} x {cols} (capped at max_features={MAX_FEATURES})")
print(f"Non-zero entries: {X_train_count.nnz}, Sparsity: {sparsity:.2f}%")
full_dtm = CountVectorizer().fit_transform(train_text)
fr, fc = full_dtm.shape
print(f"Full (uncapped) DTM: {fr} x {fc}, sparsity {(1 - full_dtm.nnz / (fr * fc)) * 100:.2f}%")

# 1f. report answer

# 2. Train and evaluate an MLP classifier

# Helper functions
def train_mlp(X, y):
    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", learning_rate_init=0.001, batch_size=256, max_iter=20, early_stopping=True, validation_fraction=0.1, n_iter_no_change=3, random_state=SEED)
    mlp.fit(X, y)
    return mlp

def evaluate(mlp, X, y_true):
    pred = mlp.predict(X)
    return (pred, accuracy_score(y_true, pred), precision_score(y_true, pred, average="macro"), confusion_matrix(y_true, pred))

def cm_frame(cm):
    return pd.DataFrame(cm, index=[f"True {n}" for n in CLASS_NAMES], columns=[f"Pred {n}" for n in CLASS_NAMES])

t0 = time.time()
mlp = train_mlp(X_train_count, train_label)
print(f"\nMLP trained in {time.time() - t0:.0f}s; epochs run: {mlp.n_iter_}, final loss: {mlp.loss_:.4f}")
print(f"Settings: hidden=(128,), relu, adam, lr=0.001, batch=256, max_iter=20, early_stopping (10% val, patience 3), loss=cross-entropy, seed={SEED}")

pred, acc, prec, cm = evaluate(mlp, X_test_count, test_label)
print(f"\nTest accuracy: {acc:.4f}, macro-averaged precision: {prec:.4f}")
print(cm_frame(cm))

print("\nThree misclassified test examples:")
for i in np.where(pred != test_label)[0][:3]:
    print(f"\nTrue: {CLASS_NAMES[test_label[i]]} | Predicted: {CLASS_NAMES[pred[i]]}\n{test_text[i]}")

# 3. Compare feature representations and stop-word removal
rng = np.random.RandomState(SEED)
sample_idx = np.concatenate([rng.choice(np.where(test_label == c)[0], 50, replace=False) for c in range(4)])
np.save("sample_idx_200.npy", sample_idx)
select_mask = np.ones(len(test_label), dtype=bool)
select_mask[sample_idx] = False

configs = [("Count",False), ("Tfidf", False), ("Count", True), ("Tfidf", True)]
runs = {("Count", False): (count_vec, mlp, pred, cm)}
rows_out = []
for kind, stop in configs:
    key = (kind, stop)
    if key not in runs:
        vec = make_vectorizer(kind, stop)
        Xtr = vec.fit_transform(train_text)
        Xte = vec.transform(test_text)
        m = train_mlp(Xtr, train_label)
        p, _, _, c = evaluate(m, Xte, test_label)
        runs[key] = (vec, m, p, c)
    vec, m, p, c = runs[key]
    rows_out.append({
        "Vectorizer": f"{kind}Vectorizer", "Stop-word removal": "Yes" if stop else "No",
        "Accuracy": accuracy_score(test_label, p),
        "Macro-precision": precision_score(test_label, p, average="macro"),
        "Acc (excl. 200-sample)": accuracy_score(test_label[select_mask], p[select_mask]),
    })

results = pd.DataFrame(rows_out)
print("\n", results.round(4).to_string(index=False))
results.to_csv("part1_results.csv", index=False)

best = int(results["Acc (excl. 200-sample)"].idxmax())
best_key = configs[best]
best_vec, best_mlp, best_pred, best_cm = runs[best_key]
print(f"\nBest configuration: {best_key[0]}Vectorizer, stop-words removed: {best_key[1]}")
print(cm_frame(best_cm))

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(best_cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=30)
ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (best config)")
for (r, c), v in np.ndenumerate(best_cm):
    ax.text(c, r, v, ha="center", va="center")
plt.tight_layout(); plt.savefig("confusion_matrix_best.png", dpi=150)

joblib.dump({"vectorizer": best_vec, "model": best_mlp, "class_names": CLASS_NAMES}, "best_topic_classifier.joblib")