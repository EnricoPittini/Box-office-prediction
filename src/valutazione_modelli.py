# FUNZIONI PER LA VALUTAZIONE DEI MODELLI.


import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler



################# VALUTAZIONE MODELLI CON TRAINING/VALIDATION/TEST

# Funzione che calcola lo score di un dato modello rispetto al training set, validation set e test set.
# In input si passa il modello e il dataset (X,y). Posso specificare se voglio scalare le features in X con MinMaxScaler. Si può specificare inoltre la
# size del test set, il random state per splittare in training e test, il numero di fold per la cross validation. Infine si può specificare se si tratta
# di regressione o classificazione.
# X ha dimenzione (n_istanze,n_features) ; y ha dimensione (n_istanze,).
def compute_train_val_test(model ,X ,y ,scale=False ,test_size=0.2 ,random_state=123, cv=5 ,regr=True):

    scoring=""
    if regr:
        scoring="neg_mean_squared_error"
    else:
        scoring="accuracy"

    if(scale): # Scalo le features in X
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # Splitto in training e test.
    X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Cross validation
    scores = cross_val_score(model, X_train_80, y_train_80, cv=cv, scoring=scoring)
    val_acc = scores.mean() # score sul validation
    if regr:
        val_acc = -val_acc

    model.fit(X_train_80,y_train_80) # Fitto usando tutto il training.

    # Calcolo la score sul training e sul test.
    train_acc=0
    test_acc=0
    if regr:
        train_acc = mean_squared_error(y_true=y_train_80, y_pred=model.predict(X_train_80))
        test_acc = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))
    else:
        train_acc = accuracy_score(y_true=y_train_80, y_pred=model.predict(X_train_80))
        test_acc = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))

    return train_acc, val_acc, test_acc # Ritorno una tripla : score sul training, validation, test.


# Funzione che data una lista modelli calcola per ciascuno di essi lo score su training/validation/test. Ritorna una lista dove per ogni modello è
# appunto calcolato lo score su training/validation/test (tripla) e ritorna anche l'indice del modello migliore : il modello migliore è quello con score
# migliore rispetto al validation.
# Alla funzione passo la lista di modelli da valutare e il dataset completo (X,y). Posso inoltre passare:
#                               - scale, test_size, random_state, cv, regr --> come spiegato in precedenza
#                               - plotta --> specifica alla funzione se deve fare il grafico sulla valutazione dei modelli
#                               - plottaTrain --> specifica se nel grafico si vuole mostrare anche lo score dei vari modelli sul training
#                               - plottaTest --> specifica se nel grafico si vuole mostrare anche lo score dei vari modelli sul test set
#                               - xvalues --> specifica i valori da mettere sull'asse delle x
#                               - xlabel --> specifica l'etichetta da mettere sull'asse x
#                               - title --> specifica il titolo da mettere al grafico
def model_selection_TrainValTest(model_list, X, y, scale=False, test_size=0.2, random_state=123, cv=5, regr=True, plotta=False, plottaTrain=False,
                                         plottaTest=False, xvalues=None, xlabel="Complessità", title="Valutazione modelli con Training/Validation/Test"):

    if(scale): # Scalo le features in X, se specificato.
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # Lista trainValTest_list : conterrà per ogni modello la tripla degli score su training/validation/test.
    trainValTest_list = []

    # Calcolo i vari score per ogni modello.
    for model in model_list:
        trainValTest_list.append(list(compute_train_val_test(model ,X, y, False, test_size, random_state, cv, regr=regr)))

    trainValTest_list = np.array(trainValTest_list) # in numpy

    if(plotta): # Faccio il grafico

        if(xvalues is None): # Valori di deafult sull'asse delle x
            xvalues = range(len(model_list))

        fig, ax = plt.subplots(figsize=(6,6))

        if plottaTrain: # Devo disegnare anche lo score sul training set.
            ax.plot(xvalues,trainValTest_list[:,0], 'o:', label='Train')
        ax.plot(xvalues,trainValTest_list[:,1], 'o:', label='Validation') # Score validation
        if plottaTest: # Devo disegnare anche lo score sul test set.
            ax.plot(xvalues,trainValTest_list[:,2], 'o:', label='Test')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid()
        ax.legend()

    # Ritorno una coppia : la lista di score train/val/test per ogni modello ; l'indice del modello con score sul validation migliore.
    if regr: # regressione
        return trainValTest_list, np.argmin(trainValTest_list,axis=0)[1]
    return trainValTest_list, np.argmax(trainValTest_list,axis=0)[1] # classificazione



########## VALUTAZIONE MODELLI CON BIAS/VARIANCE/ERROR

# Funzione che calcola il bias, la varianza e l'errore di un dato modello.
# In input si passa il modello, il dataset su cui si vuole effettuare la valutazione (X,y). Posso anche specificare se le features in X devono essere
# scalate usando MinMaxScaler. E si possono anche specificare il numero di test da fare per calcolare bias/variance/error e la dimensione di ogni
# campione rispetto al dataset completo.
# X ha dimenzione (n_istanze,n_features) ; y ha dimensione (n_istanze,).
def compute_bias_variance_error(model ,X ,y ,scale=False ,N_TESTS = 20 ,sample_size=0.67):

    # Scalo X se specificato
    if(scale):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # Vettore "vector_ypred": alla fine sarà una matrice con tante righe quanto N_TESTS (ogni riga corrisponde ad un campione) e tante colonne quanto
    # ogni punto di X (ogni colonna è un punto del dataset).
    # Riga i ---> ci sono le predizioni fatte dal modello sul campione i su tutti i punti del dataset
    # Colonna j ---> ci sono le predizioni fatte sul punto j da tutti gli N_TESTS campioni.
    vector_ypred = []

    # Itero su N_TESTS. Ad ogni iterazione estraggo il modello dallo specifico campione i.
    for i in range(N_TESTS):
        # Prendo un campione del dataset.
        Xs, ys = resample(X,y, n_samples=int(sample_size*len(y)) )

        # Estraggo il modello dal campione i.
        model.fit(Xs,ys)

        # Aggiungo le predizioni fatte dal modello su tutti i punti del dataset.
        vector_ypred.append(list(model.predict(X)))

    vector_ypred = np.array(vector_ypred) # Trasformo in numpy

    # Vettore che ha tanti elementi quanti i punti del dataset e per ciascuno ha il relativo bias calcolato sugli N_TESTS campioni.
    vector_bias = (y - np.mean(vector_ypred, axis=0))**2

    #Vettore che ha tanti elementi quanti i punti del dataset e per ciascuno ha la relativa variance calcolata sugli N_TESTS campioni.
    vector_variance = np.var(vector_ypred, axis=0)

    # Vettore che ha tanti elementi quanti i punti del dataset e per ciascuno ha il relativo error calcolato sugli N_TESTS campioni.
    vector_error = np.sum((vector_ypred - y)**2, axis=0)/N_TESTS

    bias = np.mean(vector_bias) # Bias complessivo del modello.
    variance = np.mean(vector_variance) # Variance complessiva del modello.
    error = np.mean(vector_error) # Error complessivo del modello.

    return bias,variance,error # Ritorno una tripla.


# Funzione che calcola Bias/Variance/Error per ogni modello nella lista di modelli ricevuta in input. Oltre a ritornare la lista di Bias/Variance/Error
# calcolate su ogni modello, ritorna l'indice del modello con Error minore (è il modello migliore).
# In input si passa la lista di modelli e il dataset su cui effettuare la valutazione (X,y). Inoltre si possono passare volendo altri parametri.
#                   - scale, N_TESTS, sample_size --> già descritti in precedenza
#                   - plotta --> se True, dice alla funzione di fare anche un grafico della valutazione dei modelli.
#                   - xvalues --> valori da mettere nell'asse x
#                   - xlabel --> etichetta da mettere sull'asse x
#                   - title --> titolo da dare al grafico.
def model_selection_BiasVarianceError(model_list, X, y, scale=False ,N_TESTS = 20 ,sample_size=0.67 ,plotta=False, xvalues=None, xlabel="Complessità"
                                            ,title="Valutazione modelli con Bias/Variance/Error"):
    # Scalo X se specificato
    if(scale):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # Lista biasVarianceError_list : conterrà per ogni modello la tripla Bias/Variance/Error.
    biasVarianceError_list = []

    # Calcolo Bias/Variance/Error per ogni modello.
    for model in model_list:
        biasVarianceError_list.append(list(compute_bias_variance_error(model ,X ,y ,False ,N_TESTS ,sample_size)))

    biasVarianceError_list = np.array(biasVarianceError_list) # trasformo in numpy

    # Disegno il grafico se specificato
    if(plotta):

        if(xvalues is None):
            xvalues = range(len(model_list))

        fig, ax = plt.subplots(figsize=(6,6))

        ax.plot(xvalues,biasVarianceError_list[:,0], 'o:', label='Bias$^2$')
        ax.plot(xvalues,biasVarianceError_list[:,1], 'o:', label='Variance')
        ax.plot(xvalues,biasVarianceError_list[:,2], 'o:', label='Error')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid()
        ax.legend()

    # Ritorno una coppia : la lista di Bias/Variance/Erro per ogni modello e l'indice del modello con Error minimo.
    return biasVarianceError_list, np.argmin(biasVarianceError_list,axis=0)[2]
