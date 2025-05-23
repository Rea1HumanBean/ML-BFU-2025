import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('tkagg')

path = os.path.join('..', 'data', 'diabetes.csv')
df = pd.read_csv(path)


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_proba_log = log_reg.predict_proba(X_test)[:, 1]


tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
y_proba_tree = tree.predict_proba(X_test)[:, 1]


depths = range(1, 21)
recall_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    recall_scores.append(recall)


plt.figure(figsize=(10, 6))
plt.plot(depths, recall_scores, marker='o', linestyle='-', color='b')
plt.title('Зависимость Recall от глубины решающего дерева')
plt.xlabel('Глубина дерева (max_depth)')
plt.ylabel('Recall')
plt.grid(True)
plt.xticks(depths)
plt.show()