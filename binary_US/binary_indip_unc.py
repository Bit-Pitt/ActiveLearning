import numpy as np
import pandas as pd
from ebu import *
import numpy as np



#formula dell'entropia binaria, uso di numpy per il log
#  Clip per evitare log(0) o log(1), che darebbero inf/nan. Si forza p ∈ [1e-10, 1 - 1e-10]
# Numpy permette in AUTOMATICO di effettuare questa operazione per ogni cella della matrice
def entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)                    
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


#Seleziona il sample più incerto in base alla media delle entropie.
#  model.predict_proba()         
#[   
#    [  [p0,p1] , [0.2, 0.8], ...],  # label 1 → probabilità per ogni sample
#    [[0.7, 0.3], [0.1, 0.9], ...],  # label 2
#    [[0.5, 0.5], [0.3, 0.7], ...],  # label 3
#   ...
#]   1° sample ha p1 0.3 per la label2 e p0 0.5 per label 3 ...  
def select_most_uncertain(model, X_pool):
   
    prob_per_label = model.predict_proba(X_pool)
   
    # Estrai solo la probabilità della classe positiva (label = 1) per ogni label , sono array di tipo numpy
    # Risultato: una matrice (n_samples, n_labels)       [i,j] == prob positiva del sample i per la label j (p1)
    positive_probs = np.array([label_probs[:, 1] for label_probs in prob_per_label]).T                    #.T traspone!

    sample_entropies = entropy(positive_probs)

    # Calcola la media delle entropie per ogni sample (riga)
    average_entropy_per_sample = sample_entropies.mean(axis=1)

    # Trova l'indice del sample con la maggiore incertezza media
    most_uncertain_index = np.argmax(average_entropy_per_sample)

    print("Sample with Max entropy:", np.max(average_entropy_per_sample))
    print("Mean entropy samples:", np.mean(average_entropy_per_sample))

    return most_uncertain_index

#versione in cui selezioni k sample ad ogni iterazione
def select_k_most_uncertain(model, X_pool, k=3):

    prob_per_label = model.predict_proba(X_pool)
    positive_probs = np.array([label_probs[:, 1] for label_probs in prob_per_label]).T  
    sample_entropies = entropy(positive_probs)
    average_entropy_per_sample = sample_entropies.mean(axis=1)

    # Trova gli indici dei sample con maggiore incertezza (top-k)
    top_k_indices = np.argsort(-average_entropy_per_sample)[:k]  # ordina decrescente

    print("Top entropies:", average_entropy_per_sample[top_k_indices])
    print("Mean entropy of pool:", np.mean(average_entropy_per_sample))

    return top_k_indices

#Nota:  y_pool in contesto reale non si hanno, ma noi lo sfruttiamo solo per l'etichettatura automatica (senza rumore) 
#  @algorithm: Ad ogni iterazione: 
#   - calcolo l'incertezza e ottengo il sample con max incertezza, lo inserisco nel train set e lo tolgo dal pool , riaddestro il modello
#  @return il modello finale e il nuovo train e pool set.
def active_learning(model, X_train, y_train, X_pool, y_pool, iterations=100):
    model.fit(X_train, y_train)
    for i in range(iterations):
        print(f"\n=== Iterazione {i+1}/{iterations} ===")

        # Trova l'indice del sample più incerto
        idx = select_most_uncertain(model,X_pool)
        
        X_train = pd.concat([X_train, X_pool.iloc[[idx]]])
        y_train = pd.concat([y_train, y_pool.iloc[[idx]]])

        X_pool = X_pool.drop(X_pool.index[idx])
        y_pool = y_pool.drop(y_pool.index[idx])

        model.fit(X_train, y_train)

    return model, X_train, y_train, X_pool, y_pool


#versione con selezione di k sample , e ebu
#  model.named_steps['vectorizer'] per ottenere il TfidfVectorizer (del primo passo), .transform(X_pool) trasforma il testo in valori reali (tf*idf)
# > 0.01 fa si che se il valore è maggiore allora "True" .. , infine astype(float) converte True in 1.0 e falso 0.0 ==> la matrice è pronta all'ebu   (esempio nella documentazione)
def active_learning(model, X_train, y_train, X_pool, y_pool, iterations=100, k=3,k_ebu = 10,ebu=False):
    model.fit(X_train, y_train)
    for i in range(iterations):
        print(f"\n=== Iterazione {i+1}/{iterations} ===")

        indices = select_k_most_uncertain(model, X_pool, k)   

        if ebu:
            X_pool_bin = (model.named_steps['vectorizer'].transform(X_pool) > 0.01).astype(float).toarray()   #trasformi il pool in binario (x=x1,x2,x3...)
            indices = select_by_ebu_multilabel(model, X_pool, X_pool_bin, indices, batch_size=k_ebu)

        X_train = pd.concat([X_train, X_pool.iloc[indices]])
        y_train = pd.concat([y_train, y_pool.iloc[indices]])

        X_pool = X_pool.drop(X_pool.index[indices])
        y_pool = y_pool.drop(y_pool.index[indices])

        print("Sto ritrainando il modello")
        model.fit(X_train, y_train)

    return model, X_train, y_train, X_pool, y_pool

    

def random_select(X_pool, k):
    return np.random.choice(len(X_pool), size=k, replace=False)



#Versione selezione random
def random_active_learning(model, X_train, y_train, X_pool, y_pool, iterations=10,k=50):
    model.fit(X_train, y_train)
    for i in range(iterations):
        print(f"\n=== Iterazione {i+1}/{iterations} ===")

        indices = random_select(X_pool, k)   
        print(f"Indici random:{indices}")

        X_train = pd.concat([X_train, X_pool.iloc[indices]])
        y_train = pd.concat([y_train, y_pool.iloc[indices]])

        X_pool = X_pool.drop(X_pool.index[indices])
        y_pool = y_pool.drop(y_pool.index[indices])

        model.fit(X_train, y_train)

    return model, X_train, y_train, X_pool, y_pool









#Questo è una piccola demo
'''
import numpy as np

def entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)                    
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
            
            #2 sample 3 etichette
prob_per_label = np.array([   
                [[0.1 ,0.9], [0.2, 0.8]],  
                [[0.7, 0.3], [0.1, 0.9]],  
                [[0.5, 0.5], [0.3, 0.7]],  
    ] )
    
positive_probs = np.array([label_probs[:, 1] for label_probs in prob_per_label])

print(positive_probs)
print("\n")

positive_probs = positive_probs.T #.T traspone

print(positive_probs)
print("\n")

sample_entropies = entropy(positive_probs)

print(sample_entropies)
print("\n")


average_entropy_per_sample = sample_entropies.mean(axis=1)

print(average_entropy_per_sample)
print("\n")

most_uncertain_index = np.argmax(average_entropy_per_sample)
print(most_uncertain_index)

'''