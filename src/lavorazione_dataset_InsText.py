# FUNZIONI PER LA LAVORAZIONE DEL DATASET : FEATURES INSIEMISTICHE/TESTUALI

import pandas as pd
import numpy as np

from textblob import TextBlob
from nltk.corpus import stopwords


#################### FEATURES INSIEMISTICHE

###### 1) GENRES

# Alternativa 1

# Prendo i k generi più numerosi e creo le relative feature dummy. Tutti gli altri generi li accorpo in "other_genre"
def add_genre_1(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Puliamo la colonna "genres". Per ora è una colonna di oggetti dove ogni oggetto è una stringa di lista di dizionari.
    # Togliamo i valori nulli
    df["genres"] = df["genres"].fillna("{}")
    # Ogni elemento lo rendiamo lista di dizionari
    df["genres"]=df["genres"].map(lambda obj : eval(obj))
    # Ogni lista di dizionari lo rendiamo un set
    df["genres"] = df["genres"].map(lambda lst : {dct['name'] for dct in lst})

    # Lista di tutti i generi.
    generi = set([])
    for film in list(df.index.values):
        generi |= df.loc[film,"genres"]
    generi = list(generi)

    numFilm_per_genre = [ (df[[ gen in s for s in df["genres"] ]]).shape[0] for gen in generi] # Numero film per ogni genere
    genres_sorted = np.argsort(numFilm_per_genre)[::-1] # Generi ordinati per numerosità film
    generi_selected = [ generi[gen] for gen in genres_sorted[:k] ] # Generi selezionati
    generi_in_other = set([ generi[gen] for gen in genres_sorted[k:] ]) # Generi accorpati in other
    for gen in generi_selected: # Creo le dummy per i generi selezionati
        df[gen] = df["genres"].map(lambda s : gen in s).astype(int)
    df["other_genre"] = df["genres"].map(lambda s : generi_in_other&s!=set([])).astype(int) # Creo la dummy per other_genre

    feat_list.extend(generi_selected) # Features selezionate
    feat_list.append("other_genre")


    return df[feat_list]

# Alternativa 2

# Sia $n$ il numero totale di generi. Creo $n/k$ livelli, con n mod k = 0. Nel primo metto i primi $k$ generi rispetto al revenue medio ; nel secondo
# livello metto i successivi $k$ generi ; ... ; nel k-esimo livello metto gli ultimi $k$ generi. Ho dunque $n/k$ livelli --> $n/k$ nuove features dummy.
def add_genre_2(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Puliamo la colonna "genres". Per ora è una colonna di oggetti dove ogni oggetto è una stringa di lista di dizionari.
    # Togliamo i valori nulli
    df["genres"] = df["genres"].fillna("{}")
    # Ogni elemento lo rendiamo lista di dizionari
    df["genres"]=df["genres"].map(lambda obj : eval(obj))
    # Ogni lista di dizionari lo rendiamo un set
    df["genres"] = df["genres"].map(lambda lst : {dct['name'] for dct in lst})

    # Lista di tutti i generi.
    generi = set([])
    for film in list(df.index.values):
        generi |= df.loc[film,"genres"]
    generi = list(generi)

    mean_rev_per_genre = [ (df["revenue"][ [ gen in s for s in df["genres"] ] ]).mean() for gen in generi] # Revenue medio per genere
    best_genres = np.argsort(mean_rev_per_genre)[::-1] # generi ordinati per revenue medio
    # Creo la lista dei livelli di generi : ogni livello è un insieme di generi.
    # Ogni livello è l'insieme dei k generi migliori presi fino a quel momento.
    generi_cat = [ set([generi[gen] for gen in best_genres[bound-k:bound]]) for bound in range(k,len(generi)+1,k)]
    for i,gen in enumerate(generi_cat): # Per ogni livello creo la feature dummy relativa
        df["genre_group_"+str(i+1)]=df["genres"].map(lambda s : gen&s!=set([])).astype(int)

        feat_list.append("genre_group_"+str(i+1))


    return df[feat_list]


###### 2) PRODUCTION_COMPANIES

# Prendo le k case produttrici più numerose e creo le relative feature dummy.
def add_comp(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Puliamo la colonna "production_companies". Per ora è una colonna di oggetti dove ogni oggetto è una stringa di lista di dizionari.
    # Togliamo i valori nulli
    df["production_companies"] = df["production_companies"].fillna("{}")
    # Ogni elemento lo rendiamo lista di dizionari
    df["production_companies"]=df["production_companies"].map(lambda obj : eval(obj))
    # Ogni lista di dizionari lo rendiamo un set
    df["production_companies"] = df["production_companies"].map(lambda lst : {dct['name'] for dct in lst})

    # Lista di tutti le case produttrici
    companies = set([])
    for film in list(df.index.values):
        companies |= df.loc[film,"production_companies"]
    companies = list(companies)

    numFilm_per_company = [ (df[[ c in s for s in df["production_companies"] ]]).shape[0] for c in companies] # Numerosità film per casa produttrice
    companies_sorted = np.argsort(numFilm_per_company)[::-1] # Case produttrici ordinati per numerosità film
    companies_selected = [ companies[c] for c in companies_sorted[:k] ] # Case produttrici selezionate
    for c in companies_selected: # Creo le features dummy
        df[c] = df["production_companies"].map(lambda s : c in s).astype(int)

    feat_list.extend(companies_selected) # Features selezionate


    return df[feat_list]



###### 3) KEYWORDS

# Prendo le k parole chiave più numerose e creo le relative feature dummy.
def add_key(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Puliamo la colonna "Keywords". Per ora è una colonna di oggetti dove ogni oggetto è una stringa di lista di dizionari.
    # Togliamo i valori nulli
    df["Keywords"] = df["Keywords"].fillna("{}")
    # Ogni elemento lo rendiamo lista di dizionari
    df["Keywords"]=df["Keywords"].map(lambda obj : eval(obj))
    # Ogni lista di dizionari lo rendiamo un set
    df["Keywords"] = df["Keywords"].map(lambda lst : {dct['name'] for dct in lst})

    # Lista di tutte le parole chiave
    keywords = set([])
    for film in list(df.index.values):
        keywords |= df.loc[film,"Keywords"]
    keywords = list(keywords)

    numFilm_per_keyword = [ (df[[ c in s for s in df["Keywords"] ]]).shape[0] for c in keywords] # Numerosità film per keyword
    keywords_sorted = np.argsort(numFilm_per_keyword)[::-1] # Parole chiave ordinate per numerosità film
    keywords_selected = [ keywords[c] for c in keywords_sorted[:k] ] # Parole chiave selezionate
    for c in keywords_selected: # Creo le features dummy
        df[c] = df["Keywords"].map(lambda s : c in s).astype(int)

    feat_list.extend(keywords_selected) # Features selezionate


    return df[feat_list]



###### 3) CAST

# Prendo i k attori più numerosi e creo le relative feature dummy.
def add_cast(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Puliamo la colonna "cast". Per ora è una colonna di oggetti dove ogni oggetto è una stringa di lista di dizionari.
    # Togliamo i valori nulli
    df["cast"] = df["cast"].fillna("{}")
    # Ogni elemento lo rendiamo lista di dizionari
    df["cast"]=df["cast"].map(lambda obj : eval(obj))
    # Ogni lista di dizionari lo rendiamo un set
    df["cast"] = df["cast"].map(lambda lst : {dct['name'] for dct in lst})

    # Lista di tutti gli attori
    cast = set([])
    for film in list(df.index.values):
        cast |= df.loc[film,"cast"]
    cast = list(cast)

    numFilm_per_cast = [ (df[[ c in s for s in df["cast"] ]]).shape[0] for c in cast] # Numerosità film per attore
    cast_sorted = np.argsort(numFilm_per_cast)[::-1] # Attori ordinati per numerosità film
    cast_selected = [ cast[c] for c in cast_sorted[:k] ] # Attori selezionati
    for c in cast_selected: # Creo le features dummy
        df[c] = df["cast"].map(lambda s : c in s).astype(int)

    feat_list.extend(cast_selected) # Features selezionate


    return df[feat_list]



###### 4) CREW

# Prendo i k membri della crew più numerosi e creo le relative feature dummy.
def add_crew(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Puliamo la colonna "crew". Per ora è una colonna di oggetti dove ogni oggetto è una stringa di lista di dizionari.
    # Togliamo i valori nulli
    df["crew"] = df["crew"].fillna("{}")
    # Ogni elemento lo rendiamo lista di dizionari
    df["crew"]=df["crew"].map(lambda obj : eval(obj))
    # Ogni lista di dizionari lo rendiamo un set
    df["crew"] = df["crew"].map(lambda lst : {dct['name'] for dct in lst})

    # Lista di tutti i membri della crew.
    crew = set([])
    for film in list(df.index.values):
        crew |= df.loc[film,"crew"]
    crew = list(crew)

    numFilm_per_crew = [ (df[[ c in s for s in df["crew"] ]]).shape[0] for c in crew] # Numerosità film per membro della crew
    crew_sorted = np.argsort(numFilm_per_crew)[::-1] # Membri della crew ordinati per numerosità film
    crew_selected = [ crew[c] for c in crew_sorted[:k] ] # Membri della crew selezionati
    for c in crew_selected: # Creo le features dummy
        df[c] = df["crew"].map(lambda s : c in s).astype(int)

    feat_list.extend(crew_selected) # Features selezionate


    return df[feat_list]




#################### FEATURES TESTUALI

# Parole da non considerare
stopwords=set(stopwords.words('english'))

# Funzione che data una frase ne estrae la lista di lemmi
def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence).lower()
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return lemmatized_list


#### TITLE

# Prendo le k parole più numerose e creo le relative feature dummy.
def add_title(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Trasformo ogni titolo in un insieme di parole
    df["title"] = df["title"].map(lambda s : set(lemmatize_with_postag(s)) - stopwords)

    # Prendo tutte le parole
    words = set([])
    for film in list(df.index.values):
        words |= df.loc[film,"title"]
    words = np.array(list(words))

    numFilm_per_wordTitle = np.array([ (df[[ c in s for s in df["title"] ]]).shape[0] for c in words]) # Numerosità film per parola
    best_wordTitle_num = np.argsort(numFilm_per_wordTitle)[::-1] # Parole nei titoli ordinate per numerosità film

    selected_words = words[best_wordTitle_num[1:k+1]] # Parole selezionate da mettere come features dummy. Salto la prima parola ("'s")
    for w in selected_words: # Creo le features dummy
        df[w+"_title"] = df["title"].map(lambda s : w in s).astype(int)

    feat_list.extend([w+"_title" for w in selected_words]) # Features selezionate

    return df[feat_list]



#### TAGLINE

# Prendo le k parole più numerose e creo le relative feature dummy.
def add_tagline(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Missing values
    df["tagline"] = df["tagline"].fillna("")
    # Trasformo ogni titolo in un insieme di parole
    df["tagline"] = df["tagline"].map(lambda s : set(lemmatize_with_postag(s)) - stopwords)

    # Prendo tutte le parole
    words = set([])
    for film in list(df.index.values):
        words |= df.loc[film,"tagline"]
    words = np.array(list(words))

    numFilm_per_wordTagline = np.array([ (df[[ c in s for s in df["tagline"] ]]).shape[0] for c in words]) # Numerosità film per parola
    best_wordTagline_num = np.argsort(numFilm_per_wordTagline)[::-1] # Parole nei titoli ordinate per numerosità film

    selected_words = [words[best_wordTagline_num[1]]] # Parole selezionate da mettere come features dummy
    selected_words.extend(words[best_wordTagline_num[3:k+2]]) # Prendo le prime k parole, saltando "'s" e "n't"
    for w in selected_words: # Creo le features dummy
        df[w+"_tagline"] = df["tagline"].map(lambda s : w in s).astype(int)

    feat_list.extend([w+"_tagline" for w in selected_words])

    return df[feat_list]



#### OVERVIEW


## Alternativa 1

# Prendo le k parole più numerose e creo le relative feature dummy.
def add_overview_1(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Missing values
    df["overview"] = df["overview"].fillna("")
    # Trasformo ogni titolo in un insieme di parole
    df["overview"] = df["overview"].map(lambda s : set(lemmatize_with_postag(s)) - stopwords)

    # Prendo tutte le parole
    words = set([])
    for film in list(df.index.values):
        words |= df.loc[film,"overview"]
    words = np.array(list(words))

    numFilm_per_wordOverview = np.array([ (df[[ c in s for s in df["overview"] ]]).shape[0] for c in words]) # Numerosità film per parola
    best_wordOverview_num = np.argsort(numFilm_per_wordOverview)[::-1] # Parole nei titoli ordinate per numerosità film

    selected_words = words[best_wordOverview_num[1:k+1]] # Parole selezionate da mettere come features dummy. Salto la prima parola ("'s")
    for w in selected_words: # Creo le features dummy
        df[w+"_overview"] = df["overview"].map(lambda s : w in s).astype(int)

    feat_list.extend([w+"_overview" for w in selected_words])

    return df[feat_list]



# Alternativa 2

# Funzione che data la colonna "overview" e la mappa lessico->interi costruisce il vector space.
# Il vector space contiene tante righe quanti i film e tante colonne quante le parole nel lessico. Il valore dell'elemento i,j è la frequenza della
# parola j nell'overview i : tale frequenza è scalata per l'inverse document frequency.
# (I film che sono stati tolti perchè hanno revenue<=1000 hanno tutta la riga con soli 0).
def get_vector_space (overviews, lex_map):
    m = np.zeros((max(overviews.index.values)+1, len(lex_map))) # Inizializzo tutto il vector_space a 0

    for film_id in range(len(m)): # Itero su tutti i film
        if film_id not in overviews.index.values: # Se il film non esiste significa che l'ho tolto a mano precedentemente (quando ho pulito revenue)
            continue # Per i film non esistenti tengo 0.0
        # Se il film non è stato rimosso, allora vado a prendere la sua lista di parole e riporto la frequenza di ogni parola nel vector space
        for lem in overviews.loc[film_id]:
            if lem in lex_map:
                term_id = lex_map[lem]
                m[film_id,term_id] += 1.0

    # Scalo con inverse document frequency : idf
    Ndocs = len(m)
    doc_freq = np.sum(m>0,axis=0)
    return m * np.log( Ndocs/doc_freq )

# Aggiungo le k parole più importanti rispetto al vector space. Aggiungo proprio le relative colonne del vector space.
def add_overview_2(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Missing values
    df["overview"] = df["overview"].fillna("")
    # Trasformo ogni titolo in una lista di parole
    df["overview"] = df["overview"].map(lambda s : lemmatize_with_postag(s))

    # Costruisco il lessico
    lexicon = set([])
    for film in df.index.values:
        lexicon |= set(df.loc[film,"overview"])
        lexicon = lexicon - stopwords
    lexicon_map     = {term:id for id,term in enumerate(lexicon)} # Mappa parole->id interi
    lexicon_rev_map = {id:term for term,id in lexicon_map.items()} # Mappa id interi->parole

    # Costruisco il vector space
    vector_space = get_vector_space(df["overview"],lexicon_map)

    importance_per_word = [np.sum(vector_space[:,word]) for word in range(len(lexicon))] # Importanza di ogni parola (somma della colonna del vector space)
    sorted_words = np.argsort(importance_per_word)[::-1] # Parole ordinate per importanza
    selected_words = [ lexicon_rev_map[id] for id in sorted_words[1:k+1]] # Parole selezionate : prime k parole più importanti. Tolgo la prima.

    for w in selected_words: # Per le parole selezionate costruisco le relative features : non sono altro che proprio le colonne del vector space
        df[w+"_overview"] = pd.Series(vector_space[:,lexicon_map[w]]) # Colonna del vector space di tale parola


    feat_list.extend([w+"_overview" for w in selected_words])

    return df[feat_list]



## Alternativa 3

# Aggiungo come feature il revenue medio dei k film più simili. Similarità rispetto al solo overview
def add_overview_3(dataframe,k):

    # Copia del datframe (parto con il dataframe grezzo)
    df = dataframe.copy()

    # features da selezionare
    feat_list = []

    # Prendo la matrice delle similarità dal file csv
    similarity_vector = pd.read_csv("similarity_vector.csv").values

    # Per ciascun film creo la lista dei 3*k film più simili. Lista di liste.
    # (Tengo solo 3*k film e non tutti per motivo di efficenza)
    mst_sim_films = [np.argsort(similarity_vector[film,:])[::-1][:3*k] for film in range(similarity_vector.shape[0])]

    # Filtro tenendo solo film che non ho tolto da "dataframe" (all'inizio ho tolto da "dataframe" i film con revenue<=1000)
    mst_sim_films = [[sim_film for sim_film in mst_sim_films[film] if sim_film in df.index.values] \
                           for film in range(similarity_vector.shape[0])]

    # Filtro tenendo solo k film
    mst_sim_films = [mst_sim_films[film][:k] for film in range(similarity_vector.shape[0])]

    # Per ogni film prendo la media dei revenue dei film più simili.
    mean_rev_mstSimFilms = np.array([np.mean(np.array([df.loc[sim_film,"revenue"] for sim_film in mst_sim_films[film]])) \
                                        for film in range(similarity_vector.shape[0])])

    # Metto tale vettore come nuova colonna
    df["meanRev_mstSimOverview"] = pd.Series(mean_rev_mstSimFilms)

    feat_list.append("meanRev_mstSimOverview")

    return df[feat_list]
