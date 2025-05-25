#Questo script serve per test veloci
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import random

from tqdm.auto import tqdm
from binary_indip_unc import * 
from utils import *
from models import *
from evaluation import *
from load_dataset import * 


tqdm.pandas()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sns.set_theme(style="darkgrid", palette="pastel")



SEED = 48
random.seed(SEED) 
np.random.seed(SEED)

ds_name = "Pubmed_train2.csv"
ds,TARGET_COLS = load_dataset(ds_name)
dtrain,X_train,y_train,X_pool,y_pool,X_test,y_test = ds_split(ds,500,TARGET_COLS,SEED)

print(f"Train set contains {len(X_train)} samples")
print(f"Test set contains {len(X_test)} samples")
print(f"Pool set contains {len(X_pool)} samples")
    

                # MODEL AND TRAINING

#Scelta del modello 
model = choose_model(2,SEED)

#training
model.fit(X_train, y_train)

#Evaluation senza Active Learning
evaluation(model,X_test,y_test,False)

#------------------------------------------------------------------------------------------------
#           Training with  Active Learning
#------------------------------------------------------------------------------------------------


#model.fit(X_train, y_train)          #fatto già sopra

#model, X_train, y_train, X_pool, y_pool = active_learning(model, X_train, y_train, X_pool, y_pool, 100)
model, X_train, y_train, X_pool, y_pool = active_learning(model, X_train, y_train, X_pool, y_pool, iterations=10,k=50)  
#model, X_train, y_train, X_pool, y_pool = active_learning(model, X_train, y_train, X_pool, y_pool, iterations=100,k=30,ebu=True)      

print(f"\nControllo di consistenza == Train set: {len(X_train)} | Pool set: {len(X_pool)}")

# evaluation  dopo Active Learning 
evaluation(model,X_test,y_test,False)



# -------------------------------------------------- mostra grafici
#plt.show()

 



