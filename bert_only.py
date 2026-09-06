import matplotlib
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_features, load_labels

# Load features
X_train = load_features("X_train_distilbert.pt")
X_test = load_features("X_test_distilbert.pt")
y_train, y_test = load_labels()

# Train classifier
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train.numpy(), y_train)

# Predict
y_pred = clf.predict(X_test.numpy())

# Results
print("BERT ONLY RESULTS")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["benign", "malware"],
            yticklabels=["benign", "malware"])

plt.title("BERT Only – Confusion Matrix")
plt.savefig("bert_confusion_matrix.png", dpi=300)
plt.close()

print("Accuracy:", accuracy_score(y_test, y_pred))
