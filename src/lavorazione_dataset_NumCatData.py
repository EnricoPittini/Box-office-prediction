# FUNZIONI PER LA LAVORAZIONE DEL DATASET : FEATURES NUMERICHE/CATEGORICHE/DATA


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


######## FEATURES NUMERICHE

# Prima lettura e pulizia feature numeriche.
def cleaning_data_numeric():

    data_url = "train.csv"
    dataframe = pd.read_csv(data_url) # datframe pandas

    dataframe = dataframe[dataframe["revenue"]>1000] # Tolgo revenue non significativo

    # Copio datframe grezzo
    df = dataframe.copy()

    # Trasformo il tipo di "revenue"
    df["revenue"] = df["revenue"].astype(float)

    # Castiamo a tipo int il tipo della colonna "id"
    df["id"] = df["id"].astype(int)

    # Passiamo ora alla lavorazione ed estrazione delle features numeriche.

    # Lista delle features da estrarre dal datframe e mettere nel modello
    feat_list=[]

    # BUDGET
    # Castiamo a tipo numerico (float) il tipo delle colonne che hanno come valori numeri
    df["budget"] = df["budget"].astype(float)
    # Feature dummy per i valori poco significativi
    df["budget_dummy"] = (df["budget"]<=1000).astype(int)
    # Al posto dei valori poco significativi mettiamo i valori medi.
    df["budget"] = df["budget"].map(lambda x : x if x>1000 else df["budget"].mean())

    # RUNTIME
    # Al posto dei missing values mettiamo i valori medi.
    df["runtime"].fillna(df["runtime"].mean(),inplace=True)
    # Runtime non significativo : rimpiazzo con valore medio
    df["runtime"] = df["runtime"].map(lambda x : x if x>0 else df["runtime"].mean())

    feat_list.extend(["budget","budget_dummy","popularity","runtime"]) # 4 features che seleziono


    # Estraggo vettore numpy y : vettore colonna numpy con solo "revenue".
    y = df["revenue"].values

    # Scaliamo la y con MinMaxScaler
    scaler=MinMaxScaler()
    scaler.fit(y.reshape(df.shape[0],1))
    y = scaler.transform(y.reshape(df.shape[0],1)).reshape(df.shape[0],)


    # Ritorno il datframe completo, il dataframe selzionato e lavorato, il vettore colonna numpy.
    return dataframe, df[feat_list], y



###### FEATURES CATEGORICHE

# Lavorazione di "belongs_to_collection" e "homepage". Ritorna datframe con solo tali features lavorate.
# Queste due feature vengono semplicemente trasformate in feature dummy.
def add_categorial(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # belongs_to_collection
    df["belongs_to_collection"] = (~(df["belongs_to_collection"].isna())).astype(int)
    # homepage
    df["homepage"] = (~(df["homepage"].isna())).astype(int)

    feat_list.extend(["belongs_to_collection","homepage"])

    return df[feat_list]


# Lavorazione di "original_language", alternativa 1. Ritorna dataframe con solo le feature selezionate e lavorate.
# Creo semplicemente una feature dummy rispetto al fatto se la lingua è inglese o no
def add_language_1(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # original_language : ALTERNATIVA 1
    df["original_language"] = (df["original_language"]=="en").astype(int)

    feat_list.extend(["original_language"])

    return df[feat_list]


# Lavorazione di "original_language", alternativa 2. Ritorna dataframe con solo le feature selezionate e lavorate.
# Seleziono le prime 7 lingue rispetto al revenue medio e creo le corrispondenti 7 features dummy.
def add_language_2(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # original_language : ALTERNATIVA 2
    languages = df["original_language"].unique()
    mean_revenue_per_language = [ (df["revenue"][df["original_language"]==l]).mean() for l in languages]
    best_languages = np.argsort(mean_revenue_per_language)[::-1]
    lang_selected = languages[best_languages[:7]] # lingue tenute come valori
    lang_in_other = languages[best_languages[7:]] # lingue accorpate in other
    df["original_language"].replace(lang_in_other,"other_language",inplace=True)
    df = pd.concat([df,pd.get_dummies(df["original_language"])],axis=1) # Prendo i dummies

    feat_list.extend(lang_selected) # features selezionate
    feat_list.append("other_language")

    return df[feat_list]



######### FEATURES DI TIPO DATA

# Funzione che dato un intero ritorna l'etichetta corrispondente al mese di quell'intero.
# Per semplificare la gestione degli indici dei vettori, facciamo corrispondere ai mesi gli interi da 0 a 11.
def int_to_month(n):
    s = ""
    if(n==0):
        s = "gen"
    elif(n==1):
        s = "feb"
    elif(n==2):
        s = "mar"
    elif(n==3):
        s = "apr"
    elif(n==4):
        s = "may"
    elif(n==5):
        s = "jun"
    elif(n==6):
        s = "jul"
    elif(n==7):
        s = "aug"
    elif(n==8):
        s = "sep"
    elif(n==9):
        s = "oct"
    elif(n==10):
        s = "nov"
    elif(n==11):
        s = "dec"
    else:
        raise RuntimeError("Month error: ",n)

    return s


## Alternativa 1

# Considero il mese semplicemente come una variabile categoriale a 12 livelli: da ciò 12 features dummy binarie.
def add_data_1(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # year
    df["year"] = df["release_date"].map(lambda s : int(s.split("/")[2]))
    df["year"] = df["year"].map(lambda y : 1900+y if (y>=20 and y<=99) else 2000+y)

    # month
    df["month"] = df["release_date"].map(lambda s : int(s.split("/")[0])-1) # da 0 a 11, per mapping migliore con indici

    df["month"] = df["month"].map(int_to_month)
    df = pd.concat([df,pd.get_dummies(df["month"])],axis=1) # Creo le features dummy e le aggiungo

    feat_list.append("year") # Features selezionate
    feat_list.extend([ int_to_month(m) for m in range(0,12)])

    return df[feat_list]


## Alternativa 2

# Tengo come valori distinti solo i primi 5 mesi rispetto alla numerosità di film: tutti gli altri film li accorpo nel livello "other_month". Ottengo
# quindi 6 livelli possibili --> 6 nuove features dummy.
def add_data_2(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # year
    df["year"] = df["release_date"].map(lambda s : int(s.split("/")[2]))
    df["year"] = df["year"].map(lambda y : 1900+y if (y>=20 and y<=99) else 2000+y)

    # month
    df["month"] = df["release_date"].map(lambda s : int(s.split("/")[0])-1) # da 0 a 11, per mapping migliore con indici

    numb_per_month = [ df[df["month"]==m].shape[0] for m in range(0,12)] # Numero di film per mese
    months_sorted = np.argsort(numb_per_month)[::-1] # Mesi ordinati per numerosità film
    for m in months_sorted[:5]: # Features dummy per i primi 5 mesi
        df[int_to_month(m)] = (df["month"]==m).astype(int)
    df["other_month"] = df["month"].map(lambda m : 1 if m in months_sorted[9:] else 0) # Feature "other_month"

    feat_list.append("year") # Features selezionate
    feat_list.extend([ int_to_month(m) for m in months_sorted[:5]])
    feat_list.append("other_month")

    return df[feat_list]


## Alternativa 3

# Considero come valori possibili solo i primi 5 mesi con media revenue maggiore. Tutti gli altri mesi li accorpo nel valore "other_month". Ottengo
# quindi 6 livelli possibili --> 6 nuove features dummy.
def add_data_3(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # year
    df["year"] = df["release_date"].map(lambda s : int(s.split("/")[2]))
    df["year"] = df["year"].map(lambda y : 1900+y if (y>=20 and y<=99) else 2000+y)

    # month
    df["month"] = df["release_date"].map(lambda s : int(s.split("/")[0])-1) # da 0 a 11, per mapping migliore con indici

    #numb_per_month = [ df[df["month"]==m].shape[0] for m in range(0,12)]
    mean_rev_per_month = [ (df["revenue"][df["month"]==m]).mean() for m in range(0,12)] # Revenue medio per mese
    best_months = np.argsort(mean_rev_per_month)[::-1] # Mesi ordinati per revenue medio
    for m in best_months[:5]: # Features dummy per i primi 5 mesi
        df[int_to_month(m)] = (df["month"]==m).astype(int)
    df["other_month"] = df["month"].map(lambda m : 1 if m in best_months[5:] else 0) # Feature "other_month"

    feat_list.append("year") # Features selezionate
    feat_list.extend([ int_to_month(m) for m in best_months[:5]])
    feat_list.append("other_month")

    return df[feat_list]


## Alternativa 4

# Creiamo una sola feature categoriale binaria (dummy) : 1 se il mese del film è nei primi 6 mesi con revenue maggiore ; 0 se è nei 6 con revenue
# peggiore.
def add_data_4(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # year
    df["year"] = df["release_date"].map(lambda s : int(s.split("/")[2]))
    df["year"] = df["year"].map(lambda y : 1900+y if (y>=20 and y<=99) else 2000+y)

    # month
    df["month"] = df["release_date"].map(lambda s : int(s.split("/")[0])-1) # da 0 a 11, per mapping migliore con indici

    mean_rev_per_month = [ (df["revenue"][df["month"]==m]).mean() for m in range(0,12)] # Revenue medio per mese
    best_months = np.argsort(mean_rev_per_month)[::-1] # Mesi ordinati per revenue medio
    df["month"] = df["month"].map(lambda m : 1 if m in best_months[:6] else 0) # Feature binaria

    feat_list.append("year")
    feat_list.append("month")

    return df[feat_list]


## Alternativa 5

# Funzione ausiliaria che dato un intero (mese) e la lista ordinata di mesi ritorna il livello di quel mese. 3 livelli possibili.
def transform_5(m, best_months):
    if m==best_months[0]:
        return "month_group_1"
    elif m in best_months[1:6]:
        return "month_group_2"
    elif m in best_months[6:12]:
        return "month_group_3"
    else:
        raise RuntimeError("Mese invalido: ",m)

# Dividiamo i mesi in 3 gruppi : il primo mese più rilevante ; gli altri 5 mesi più rilevanti ; i restanti 6 mesi più rilevanti.
# Feature categoriale con 3 livelli --> dunque 3 features dummy.
def add_data_5(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # year
    df["year"] = df["release_date"].map(lambda s : int(s.split("/")[2]))
    df["year"] = df["year"].map(lambda y : 1900+y if (y>=20 and y<=99) else 2000+y)

    # month
    df["month"] = df["release_date"].map(lambda s : int(s.split("/")[0])-1) # da 0 a 11, per mapping migliore con indici

    mean_rev_per_month = [ (df["revenue"][df["month"]==m]).mean() for m in range(0,12)]
    best_months = np.argsort(mean_rev_per_month)[::-1] # Mesi ordinati per revenue medio
    df["month"] = df["month"].map(lambda m : transform_5(m,best_months)) # Trasformo i mesi nei 3 livelli
    df = pd.concat([df,pd.get_dummies(df["month"])],axis=1) # Aggiungo le dummy.

    feat_list.extend(["year","month_group_1","month_group_2","month_group_3"]) # Feature selezionate.

    return df[feat_list]


## Alternativa 6

# Funzione ausiliaria che dato un intero(mese) e data la lista di film ordinata ritorna il livello di quel mese. 3 livelli possibili.
def transform_6(m, best_months):
    if m in best_months[0:4]:
        return "month_group_1"
    elif m in best_months[4:8]:
        return "month_group_2"
    elif m in best_months[8:12]:
        return "month_group_3"
    else:
        raise RuntimeError("Mese invalido: ",m)

# Dividiamo i mesi in 3 gruppi : primi 4 mesi migliori rispetto a revenue; successivi 4 mesi migliori ; ultimi 4 mesi. Dunque sempre 3 livelli, ma
# questa volta più bilanciati. Dunque abbiamo 3 livelli possibili per la features categorica mese: da ciò 3 features dummy binarie.
def add_data_6(dataframe):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # year
    df["year"] = df["release_date"].map(lambda s : int(s.split("/")[2]))
    df["year"] = df["year"].map(lambda y : 1900+y if (y>=20 and y<=99) else 2000+y)

    # month
    df["month"] = df["release_date"].map(lambda s : int(s.split("/")[0])-1) # da 0 a 11, per mapping migliore con indici

    best_months = np.argsort([df["revenue"][df["month"]==m].mean() for m in range(0,12)])[::-1] # Mesi ordinati per revenue medio
    df["month"] = df["month"].map(lambda m : transform_6(m,best_months)) # Trasformo nei 3 livelli
    df = pd.concat([df,pd.get_dummies(df["month"])],axis=1) # Aggiungo le dummy.

    feat_list.extend(["year","month_group_1","month_group_2","month_group_3"]) # Features selezionate

    return df[feat_list]
