#Lo script testa random, entropy, ebu singolarmente ad ogni iterazione
#Fai 40 iterazioni dove parti da 1000 sample, ed a ogni iterazione ottiene 1000 sample ==> genera un json:
#  [{  it: 20
#      metrics: ... 
#   }, ... ]
#


from load_dataset import carica_ds
from load_dataset import * 
import numpy as np
from binary_indip_unc import * 
from utils import *
from models import *
from evaluation import *
from tqdm.auto import tqdm
import os


#Setup per consistenza
tqdm.pandas()
path_results = os.path.abspath("results_iterative")
ds_name = "Pubmed_train2.csv"
ds,TARGET_COLS = carica_ds(ds_name)
methods = ['randomAL', 'entropyAL','ebuAL']
seed = 42
iterations = 40                                                                         #-----------
sample_per_iter = 1000
size = 1000     #size iniziale

#Scegline uno alla volta  (la computazione è lunga)
method = 'ebuAL'


model = choose_model(2,seed)
np.random.seed(seed)
dtrain,X_train,y_train,X_pool,y_pool,X_test,y_test = ds_split(ds,size,TARGET_COLS,seed)
print(f"Dimensione del pool: {len(X_pool)}")


if method == 'randomAL':
    random_active_learning(model, X_train, y_train, X_pool, y_pool, 
                           iterations=iterations,
                           k=sample_per_iter,
                           X_test=X_test,
                           y_test=y_test
                           )


elif method == 'entropyAL':
    active_learning(model, X_train, y_train, X_pool, y_pool, 
                    iterations=iterations,
                    k=sample_per_iter,
                    X_test=X_test,
                    y_test=y_test
                    )
  

elif method == 'ebuAL':
    active_learning(model, X_train, y_train, X_pool, y_pool, 
                    iterations=iterations, 
                    k=sample_per_iter*3, 
                    k_ebu=sample_per_iter, 
                    ebu=True,
                    X_test=X_test,
                    y_test=y_test
                    )
              
                    

      

