#Implementazione del metodo ebu sfruttando la implementazione standard dell'active learning per Multi-label
from models import * 
import numpy as np

#Se vuoi selezionare solo gli argmin o gli argmax basta commentare ad esempio " highest = sorted_indices[-batch_size:] "
def selections(total_score,top_k_indices ,batch_size):
    selected = []
    if batch_size == 2:
        idx_min = np.argmin(total_score)                          
        idx_max = np.argmax(total_score)

        #otteniamo dai locali i globali del pool
        selected = [top_k_indices[idx_min], top_k_indices[idx_max]]
    else:       #DEMO SOTTO
        sorted_indices = np.argsort(total_score)
        lowest = []
        highest = []

        lowest = sorted_indices[:batch_size]        # mancaza di evidenza
        #highest = sorted_indices[-batch_size:]      # I più "conflittuali" (alta evidenza)

        # Mappa gli indici locali (nei top_k) in indici globali del pool
        selected = [top_k_indices[i] for i in np.concatenate([lowest, highest])]

    return selected
    

def select_by_ebu_multilabel(model, X_pool, X_pool_bin, top_k_indices, batch_size=2):
    y_pred = model.predict(X_pool)              #matrice (n_samples, n_labels)
    n_labels = y_pred.shape[1]                      #shape[1] conti le colonne
    epsilon = 1e-6                                  # per evitare 0

    X_candidates = X_pool_bin[top_k_indices]        #ottieni i sample selezionati come più incerti
    n_samples = X_candidates.shape[0]               # == len(top_k_indices)
    total_scores = np.zeros(n_samples)              # vettore di lunghezza "n_samples" riempito con 0, ci calcoleremo le ebu

    for label_idx in range(n_labels):
    #Applichiamo la Ebu come da paper
        y_label = y_pred[:, label_idx]              # estrae (vettore colonna) quindi predizioni sono di quella label
        pos_mask = y_label == 1                     #Es:  [0,0,1,0,1] ==> [false false true false true]
        neg_mask = y_label == 0

        #p(x^m | 1) =  num_sample (classificati come 1) che hanno la feature  / totali sample classificati 1,     demo sotto
        p_x_given_1 = (X_pool_bin[pos_mask].astype(bool).sum(axis=0) + epsilon) / (pos_mask.sum() + epsilon)        #p(x^m | 1)
        p_x_given_0 = (X_pool_bin[neg_mask].astype(bool).sum(axis=0) + epsilon ) / (neg_mask.sum() + epsilon)        #p(x^m | 0)

        p_ratio = p_x_given_1 / p_x_given_0
        inv_ratio = p_x_given_0 / p_x_given_1

        # Feature masks: quali feature entrano in E1 e in E0
        E1_mask = p_ratio > 1
        E0_mask = inv_ratio > 1

        # Prendiamo solo le feature attive (== 1) e in evidenza
        E1_vals = np.where(E1_mask, p_ratio, 1.0)
        E0_vals = np.where(E0_mask, inv_ratio, 1.0)

        #Rispetto al paper il calcolo viene fatto sui logaritmi per evitare overflow 
        log_E1 = np.sum(np.where(X_candidates == 1, np.log(E1_vals), 0.0), axis=1)         #log(E1) = log(p1) + log(p2) ... = log(p1*p2...)
        log_E0 = np.sum(np.where(X_candidates == 1, np.log(E0_vals), 0.0), axis=1)
        total_scores += log_E1 + log_E0  # somma dei log = log(E1 * E0)
    
    #total_scores = total_scores / n_labels                  #se vuoi la media dei log(E1*E0)        

    return selections(total_scores,top_k_indices, batch_size)




'''   PRIMA DEMO
import numpy as np
epsilon = 1e-6
#   F1  F2  F3
m1= [[0, 1, 0],         #matrice dei sample "positivi" nel pool
     [1, 1, 0],
     [0, 1, 1]]
     
m0= [[0, 1, 0],         ##matrice dei sample "negativi" nel pool
     [1, 1, 1],
     [1, 0, 1]]

m1 = np.array(m1)     #trasformo in matrici numpy per applicarci i metodi
m0 = np.array(m0)

#print(m1.astype(bool).sum(axis=1))
print(m1.astype(bool).sum(axis=0) +epsilon )
print( (m1.astype(bool).sum(axis=0) +epsilon ) / (3+epsilon))
print("--------------------------------------------------")
print(m0.astype(bool).sum(axis=0) +epsilon )
print( (m0.astype(bool).sum(axis=0)+epsilon ) / (3+epsilon))

m1 = (m1.astype(bool).sum(axis=0) ) / (3+epsilon) 
m0 = (m0.astype(bool).sum(axis=0) ) / (3+epsilon)

print("\nRateo e inv rateo")
p_ratio = m1 / m0         # [0.5 1.5 0.5]  di fatti la feature positiva è solo F2 (>1)
inv_ratio = m0 / m1         # anche non calcolabile
print(p_ratio)
print(inv_ratio)

#mask delle feature     (diviosione in positive e negative)
print("\nMaschere delle feature")
E1_mask = p_ratio > 1
E0_mask = inv_ratio > 1
print(E1_mask)          # [false true false]                
print(E0_mask)          # [duale]    

#Quindi ora presi i sample candidati la sua E1 per questa label sarà la moltiplicazione di solo la F2 se presente
# per il calcolo di E0 invece sarà potenzialmente F1*F3 ma solo se sono presenti e poi calcoli E0*E1

print("\nCalcolo di E1 E0")
E1_vals = np.where(E1_mask, p_ratio, 1.0)   
E0_vals = np.where(E0_mask, inv_ratio, 1.0)
#Se la feature è in Ex_mask, prende il valore p_ratio altrimenti, mette 1.0 (neutro per il prodotto)
print(E1_vals)
print(E0_vals)

print("\ncandidati")
X_candidates = [[0, 1, 0],         #top k uncertain candidati
                [1, 1, 1]]
X_candidates = np.array(X_candidates)
E1 = np.prod(np.where(X_candidates == 1, E1_vals, 1.0), axis=1)
E0 = np.prod(np.where(X_candidates == 1, E0_vals, 1.0), axis=1)
print(E1)          # [1.5 1.5] ovvero la E1 è 1.5 sia per sample[0] che sample[1]
print(E0)          # [1   4  ]   ovvero F1*F3 per entrambi i sample 

print("\nTotal scores")
total_scores = np.zeros(len(X_candidates))
total_scores += E1 * E0            #sarà sommato per ogni label
print(total_scores)            # [1.5 6]   Primo sample E1*E0 = 1.5 , secondo 6

print("\n Indici")
idx_min = np.argmin(total_scores)               #quelli a maggior conflitto
idx_max = np.argmax(total_scores)               #quelli a minor conflitto
print(idx_min)
print(idx_max)
'''


'''  DEMO PER LA SELEZIONE 
import numpy as np
top_k_indices = [15,25,55,100,150]          #contine l'indice dei top 5 uncertain selezionati
total_scores = np.array([10,22,5,129,1])        #i loro score calcolati

sorted_indices = np.argsort(total_scores)
print(sorted_indices)


lowest = sorted_indices[:2]        #sono i più incerti
print(lowest)

# I più "conflittuali" (alta evidenza)
highest = sorted_indices[-2:]
print(highest)

# Mappa gli indici locali (nei top_k) in indici globali del pool
selected = [top_k_indices[i] for i in np.concatenate([lowest, highest])]
print(selected)
'''