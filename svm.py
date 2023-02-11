from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import get_occupancy_data
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = get_occupancy_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    clf = make_pipeline(StandardScaler(), SVC())

    # Modify kernel functions to determine overfitting? Or modify C
    clf.fit(X_train, y_train)
