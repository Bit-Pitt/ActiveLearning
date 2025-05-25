import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def choose_model(index,seed=42):
    model_found = False
    model = []

    if index == 0:
        model = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("svd", TruncatedSVD(n_components=256)),
            ("classifier", MultiOutputClassifier(estimator=xgb.XGBClassifier()))
        ]) 
        model_found = True

    elif index == 1:  # Logistic Regression, versione leggera
        model = Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=500)),  
            ("svd", TruncatedSVD(n_components=64)),             
            ("classifier", MultiOutputClassifier(
                estimator=LogisticRegression(
                    solver='liblinear',     
                    C=0.1,                  
                    max_iter=100
                )
            ))
        ]) 
        model_found = True

    elif index == 2:  # Random Forest
        model = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("svd", TruncatedSVD(n_components=128)),
            ("classifier", MultiOutputClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    random_state=seed
                )
            ))
        ])
        model_found = True

    elif index == 3:  # Gradient Boosting
        model = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("svd", TruncatedSVD(n_components=128)),
            ("classifier", MultiOutputClassifier(
                estimator=GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=seed
                )
            ))
        ])
        model_found = True

    if not model_found:
        raise Exception("No model found")
    return model
