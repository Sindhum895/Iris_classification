import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


url = "IRIS.csv"
iris_data = pd.read_csv(url)


print(iris_data.head())


X = iris_data.drop('species', axis=1)  # Features
y = iris_data['species']  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)


predictions = knn.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    new_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    prediction = knn.predict(new_data)
    return prediction[0]


new_flower_prediction = predict_flower(5.1, 3.5, 1.4, 0.2)
print(f"Predicted species for new flower: {new_flower_prediction}")
