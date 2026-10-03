import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("customer_churn_model_v2.csv")

X = df.drop(columns=["CustomerID", "Churn"])
y = (df["Churn"] == "Yes").astype(int)

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", MinMaxScaler(), numeric_cols),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Export customer churn risk scores
risk = X_test.copy()
risk.insert(0, "CustomerID", df.loc[X_test.index, "CustomerID"].values)
risk["ActualChurn"] = np.where(y_test.values == 1, "Yes", "No")
risk["ChurnProbability"] = probabilities
risk["RiskLevel"] = pd.cut(
    probabilities,
    bins=[-0.01, 0.33, 0.66, 1.01],
    labels=["Low", "Medium", "High"]
)
risk.sort_values("ChurnProbability", ascending=False).to_csv(
    "customer_churn_risk_predictions.csv", index=False
)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, probabilities):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Model")
plt.legend()
plt.tight_layout()
plt.savefig("churn_roc_curve.png", dpi=180)
plt.show()
