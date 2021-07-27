# COSTRUZIONE DEL SIMILARITY VECTOR USANDO FEATURE "OVERVIEW"
# Questo file python vuole mostrare come è stato costruito il file "similarity_vector.csv"


import numpy as np
import pandas as pd
from lavorazione_dataset_InsText import get_vector_space


# Lettura dataset

data_url = "train.csv"
df = pd.read_csv(data_url)


# Costruzione vector space. Uso funzione definita nel file "lavorazione_dataset_InsText"
# Il vector space contiene tante righe quanti i film e tante colonne quante le parole nel lessico. Il valore dell'elemento i,j è la frequenza della
# parola j nell'overview i : tale frequenza è scalata per l'inverse document frequency.
# (I film che sono stati tolti perchè hanno revenue<=1000 hanno tutta la riga con soli 0).

df["overview"] = df["overview"].fillna("")
df["overview"] = df["overview"].map(lambda s : lemmatize_with_postag(s))

lexicon = set([])
for film in df.index.values:
    lexicon |= set(df.loc[film,"overview"])
lexicon = lexicon - stopwords

lexicon_map     = {term:id for id,term in enumerate(lexicon)}
lexicon_rev_map = {id:term for term,id in lexicon_map.items()}

vector_space = get_vector_space(df["overview"],lexicon_map)


# Costruzione similarity vector
# Tale vettore è una matrice quadrata n*n, con n numero di film. Contiene le similarità per ogni coppia di film. Sappiamo che n=3000.
# Per gli elementi i,i (diagonale principale) non calcolo la similarità, ma metto semplicemente -1.
# Per i film che non hanno parole (tutta la riga nel vector space è formata da 0) tutti i relativi valori nel similarity_vector sono -1.

# Uso il coseno per calcolare similarità
def cosine(a,b):
    if np.sqrt(np.sum(a**2.0)) * np.sqrt(np.sum(b**2.0)) == 0.0:
        return -1
    return np.dot(a,b)/( np.sqrt(np.sum(a**2.0)) * np.sqrt(np.sum(b**2.0)) )

# Siccome ci sono moltissime righe e colonne nel vector space (3000*15880), uso il multithreading per ottimizzare la costruzione.
# Usando il multithreading tale costruzione ci ha comunque impiegato 45 minuti sul mio laptop.
from threading import Thread

# Inizializzo similarity_vector come una matrice di soli -1
similarity_vector = np.array([[-1.0 for j in range(len(vector_space))] for i in range(len(vector_space))])

# Funzione che calcola le similarità tra tutti i film con indice tra inf e sup.
# Come ottimizzazione, mi calcolo solo le similarità tra film i,j con i<j : ovvero calcolo solo gli elementi della matrice sopra la diagonale principale.
# Questo perchè tanto la similarità è simmetrica.
def compute_similarities(inf,sup):
    for film_i in range(inf,sup):
        for film_j in range(film_i+1,len(vector_space)):
            similarity_vector[film_i,film_j] = cosine(vector_space[film_i,:],vector_space[film_j,:])
    print("FINITO")

# Ora lancio 5 threads, ognuno costruisce una parte diversa del similarity vector. (Ricordo che in tutto ci sono 3000 film).
# I thred eseguono tutti la stessa funzione, modificando lo stesso similarity_vector condiviso. Ciò non crea race condition, perchè ognuno modifica
# una partizione diversa del similarity_vector.

threads = [] # Lista di threads
t1 = Thread(target=compute_similarities, args=(0,600))
t1.start()
threads.append(t1)
t2 = Thread(target=compute_similarities, args=(600,1200))
t2.start()
threads.append(t2)
t3 = Thread(target=compute_similarities, args=(1200,1800))
t3.start()
threads.append(t3)
t4 = Thread(target=compute_similarities, args=(1800,2400))
t4.start()
threads.append(t4)
t5 = Thread(target=compute_similarities, args=(2400,3000))
t5.start()
threads.append(t5)

# Faccio la join su tutti i thread.
for t in threads:
    t.join()

# Completo la matrice, mettendo valori anche nel triangolo inferiore.
for i in range(similarity_vector.shape[0]):
    for j in range(0,i):
        similarity_vector[i,j] = similarity_vector[j,i]

# Infine salvo ciò in un file csv
sim_df = pd.DataFrame(similarity_vector)
sim_df.to_csv("similarity_vector.csv",index=False)
