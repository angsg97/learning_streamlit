import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("Streamlit example")

st.write("""
# Explore Different Classifiers
Which one is the best one?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine dataset":
        data = datasets.load_wine()

    X = data.data
    Y = data.target
    
    return X, Y

X, Y = get_dataset(dataset_name)
st.write(f"Shape of Dataset: {X.shape}")
st.write(f"Number of Classes: {len(np.unique(Y))}")

def add_parameter_ui(clf_name: str) -> dict[any]:
    params = {}

    if clf_name == "KNN":
        params["K"] = st.sidebar.slider("K", 1, 15)
    elif clf_name == "SVM":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0)
    elif clf_name == "Random Forest":
        params["max_depth"] = st.sidebar.slider("max_depth", 2, 15)
        params["n_estimators"] = st.sidebar.slider("n_estimators", 1, 100)

    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name: str, params: dict[any]) -> KNeighborsClassifier | SVC | RandomForestClassifier:
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1111)

    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1111)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, Y_pred)
st.write(f"Classifier: {classifier_name}")
st.write(f"Accuracy: {acc}")

# Plot datasets
pca = PCA(2)  # Init 2D
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=Y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# instead of plt.show()
st.pyplot(fig)