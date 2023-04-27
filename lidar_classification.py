# lidar_classification.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


class LidarClassification:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.clf = None

    def train_classifier(self, n_estimators, n_jobs, random_state):
        # ...
        # Remove rows with missing values (NaN)
        self.clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)
        self.clf.fit(self.x, self.y)

    def evaluate_classifier(self, x_test, y_test):

        y_prediction = self.clf.predict(x_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_prediction))
        print("\nClassification Report:")
        print(classification_report(y_test, y_prediction))

    def predict_class_labels(self, feature_matrix_clean):
        # ...
        predicted_class_labels = self.clf.predict(feature_matrix_clean)
        return predicted_class_labels

