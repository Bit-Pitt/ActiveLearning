import os
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.auto import tqdm
from binary_indip_unc import * 
from utils import *
from models import *
from evaluation import *



#dim Train è grandezza del Train set in % sul dataset caricato
def load_dataset(ds_name):

    if ds_name == "train.csv":
        path = os.path.abspath("dataset")
        dtrain = pd.read_csv(os.path.join(path, "train.csv"))
        

        rows, columns = dtrain.shape
        print(f"The train set contains {rows} rows and {columns} columns")

        '''
        print(dtrain.head())
        print(dtrain.tail())
        print(dtrain.info())
        print(dtrain.isna().sum())
        print(dtrain.describe())
        print(dtrain.duplicated().sum())
        '''
        
        ID_COL = "ID"
        FEATURES = ["TITLE", "ABSTRACT"]
        TARGET_COLS = [col for col in dtrain.columns if col not in [ID_COL] + FEATURES]

        #text cleaning, creazione di nuove colonne con testi processati
        processor = Processor()
        processor(dtrain['TITLE'].iloc[9])
        dtrain['PROCESSED_TITLE'] = dtrain['TITLE'].progress_apply(processor)
        dtrain['PROCESSED_ABSTRACT'] = dtrain['ABSTRACT'].progress_apply(processor)
        dtrain['PROCESSED_TEXT'] = dtrain['PROCESSED_TITLE'] + " " + dtrain['PROCESSED_ABSTRACT']


        # splitto le colonne di "input" con le label
        X = dtrain['PROCESSED_TEXT']
        y = dtrain[TARGET_COLS]    
        '''
        #data splitting: splitto in  train set, test set, pool set
        #step 1: ottengo un piccolo train set
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.99)

        # Step 2: Il resto è 90% pool  e 10% test
        X_pool, X_test, y_pool, y_test = train_test_split(X_rest, y_rest, test_size=0.10)

        #rimpicciolisco ulteriormente il train       (il classificatore impara troppo bene già con pochi  esempi (tipo 200))
        #X_train, _, y_train, __ = train_test_split(X_train, y_train, test_size=0.20)
        '''
        return dtrain,TARGET_COLS
    
    if ds_name == "Pubmed_train.csv" or ds_name == "Pubmed_train2.csv":

        path = os.path.abspath("dataset")
        dtrain = pd.read_csv(os.path.join(path, ds_name))

        rows, columns = dtrain.shape
        print(f"The dataset contains {rows} rows and {columns} columns")
        #print(dtrain.columns)

        # Rinomina le colonne per usare stessa logica del precedente esempio
        dtrain.rename(columns={"Title": "TITLE", "abstractText": "ABSTRACT"}, inplace=True)

        # Identificatori
        FEATURES = ["TITLE", "ABSTRACT"]
        ID_COL = "pmid" if "pmid" in dtrain.columns else "ID"
        
        # Target: le colonne  'A' ... 'Z' (multi-label)     (tolta la "V" in quanto tutti valori 0)
        TARGET_COLS = [col for col in dtrain.columns if col in list("ABCDEFGHIJKLMNZ")]
                                                                    
        # Rimuovi righe senza etichette
        dtrain = dtrain[dtrain[TARGET_COLS].sum(axis=1) > 0]

        # Preprocessing del testo
        processor = Processor()
        dtrain['PROCESSED_TITLE'] = dtrain['TITLE'].fillna('').progress_apply(processor)
        dtrain['PROCESSED_ABSTRACT'] = dtrain['ABSTRACT'].fillna('').progress_apply(processor)
        dtrain['PROCESSED_TEXT'] = dtrain['PROCESSED_TITLE'] + " " + dtrain['PROCESSED_ABSTRACT']

        # Features e labels
        X = dtrain['PROCESSED_TEXT']
        y = dtrain[TARGET_COLS]

        #Stampa di occorrenza delle label nel data set
        print(y.sum(axis=0))
       
        return dtrain,TARGET_COLS
    
    raise Exception("Dataset non trovato")


def ds_split(ds,size_train,TARGET_COLS,seed):
    print(f"\n'ds_split' è implementata attualmente solo per ds:pubmed2")
    if size_train == 500:
        test_size = 0.99
    elif size_train == 2000:
        test_size = 0.94
    elif size_train == 10000:
        test_size = 0.80
    elif size_train == 25000:
        test_size = 0.65
    else:
        raise ValueError("Parameter size_train not in [500, 3000, 10000, 25000]")

    X = ds['PROCESSED_TEXT']
    y = ds[TARGET_COLS]

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y.sum(axis=1))
    X_pool, X_test, y_pool, y_test = train_test_split(X_rest, y_rest, random_state=seed, test_size=0.10 )

    return ds, X_train, y_train, X_pool, y_pool, X_test, y_test
