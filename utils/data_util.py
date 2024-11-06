from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def init_classifier(estimators=None):
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    return voting_clf


def feature_extraction(data_loader, n_components=100):
    data_features, data_labels = [], []
    for data in tqdm(data_loader, desc='Feature Extraction'):
        img, label = data
        data_features.append(img.view(img.size(0), -1).numpy())
        data_labels.extend(label.numpy())
    data_features = np.vstack(data_features)
    # pca = PCA(n_components=n_components)
    # data_features = pca.fit_transform(data_features)
    return data_features, data_labels

def write_results_to_file(filename, accuracy, f1, precision, recall,clf_list,train_size,test_size,time_taken):
    with open(filename, 'a') as f:
        f.write(f"Classifiers: {clf_list}\n")
        f.write(f"Train Size: {train_size}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Time Taken: {time_taken}\n")
        f.write("\n")