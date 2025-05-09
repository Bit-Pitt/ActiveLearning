
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression



def choose_model(index):
    model = []
    if index == 0:
        model = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("svd", TruncatedSVD(n_components=256)),
            ("classifier", MultiOutputClassifier(estimator=xgb.XGBClassifier()))
        ]) 
    elif index == 1:                #versione meno potente
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

    return model

    
    
        