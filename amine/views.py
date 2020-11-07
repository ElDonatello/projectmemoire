import json
from pandas.io.json import json_normalize
from django.shortcuts import render, render_to_response
import pandas as pd
import requests
import datetime
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from yellowbrick.cluster import KElbowVisualizer

articles = []
sujets = []
nlp = spacy.load("en_core_web_sm")
import re
import os
from nltk.stem.porter import *

p_stemmer = PorterStemmer()


def extract_entite_nomme(s):
    L = []
    # tout les etapes d'annotation se fait ici avec cette instruction
    article = nlp(s)
    for ent in article.ents:
        if (ent.label_ not in ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'PERCENT', 'QUANTITY']):
            for token in ent:
                if not (token.is_punct | token.is_stop):
                    # token1 = n2w(token.text) #convert chiffre vers mot
                    # token2 = replace_acronyms(token1) #convert abbreviation vers mot
                    L.append(token.text)

    return L


def home2(request):
    global articles
    global sujets

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    yesterday2 = today - datetime.timedelta(days=2)

    aujourd = '"' + str(today) + '"'
    yestday = '"' + str(yesterday) + '"'
    yestday2 = '"' + str(yesterday2) + '"'
    query = "today"
    url = "https://rapidapi.p.rapidapi.com/api/search/NewsSearchAPI"

    for date in [aujourd, yestday, yestday2]:
        print(date)
        querystring = {"pageSize": "100", "q": query, "autoCorrect": "true", "pageNumber": "1",
                       "toPublishedDate": "null", "withThumbnails": "true", "fromPublishedDate": date,
                       "safeSearch": "true"}

        headers = {
            'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
            'x-rapidapi-key': "a089200dbamshd00bb86da392cd7p19dd23jsn694f8679b489"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)
        detaille1 = json_normalize(response.json(), 'value')
        detaille = pd.DataFrame(columns=detaille1.columns)
        detaille = pd.concat([detaille, detaille1])

    detaille.reset_index(drop=True, inplace=True)
    detaille_article = detaille[~(detaille.id.duplicated())]
    detaille_article = detaille_article[~detaille_article.title.isna()]
    detaille_article = detaille_article[~(detaille_article.body.str.isspace())]

    detaille_article.loc[:, 'complet'] = detaille_article["title"] + " " + detaille_article["title"] + " " + \
                                         detaille_article['body']

    tfidf = TfidfVectorizer(tokenizer=extract_entite_nomme)
    dtm = tfidf.fit_transform(detaille_article.complet)
    x = dtm.toarray()

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 40))

    visualizer.fit(dtm)  # Fit the data to the visualizer
    visualizer.show()
    nombre_cluster = visualizer.elbow_value_
    k_means = KMeans(n_clusters=nombre_cluster, random_state=42)

    k_means.fit(dtm)

    closest, _ = pairwise_distances_argmin_min(k_means.cluster_centers_, x)
    all_data = [i for i in range(detaille_article.id.size)]

    m_clusters = k_means.labels_.tolist()

    centers = np.array(k_means.cluster_centers_)

    closest_data = []
    for i in range(nombre_cluster):
        center_vec = centers[i]
        data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(m_clusters) if clu_num == i]

        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = x[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        closest, _ = pairwise_distances_argmin_min([center_vec], one_cluster_tf_matrix)
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
        data_id = all_data[closest_data_row_num]

        closest_data.append(data_id)

    closest_data = list(set(closest_data))
    detaille_article['id_cluster'] = k_means.labels_

    entities = {}
    for k in detaille_article.groupby("id_cluster").count().id.nlargest(20).index:
        for i in range(nombre_cluster):
            if (detaille_article.loc[closest_data[i], 'id_cluster'] == k):
                doc = nlp(detaille_article.loc[closest_data[i], 'title'])
                entity = "nothing"
                nombre_entity = 0
                if not (doc.ents):
                    doc = nlp(detaille_article.loc[closest_data[i], 'body'])

                for ent in doc.ents:

                    if ((len(ent.text) > 2) & (
                            ent.label_ not in ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'PERCENT', 'QUANTITY'])):
                        if (detaille_article[detaille_article.loc[:, 'id_cluster'] == k].complet.str.contains(
                                ent.text, flags=re.IGNORECASE, regex=True).sum() > nombre_entity):
                            entity = ent.text
                            nombre_entity = detaille_article[
                                detaille_article.loc[:, 'id_cluster'] == k].body.str.contains(ent.text,
                                                                                              flags=re.IGNORECASE,
                                                                                              regex=True).sum()
                if entity != 'nothing':
                    entities[k] = entity

    detaille_article.sort_values("datePublished", axis=0, ascending=False, inplace=True)
    detaille_article.rename(columns={'image.url': 'thumbnail', 'provider.name': 'source'}, inplace=True)
    articles = detaille_article
    sujets = entities
    print(entities)
    json_records = detaille_article.reset_index().to_json(orient='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data, 'e': entities}
    return render(request, 'html2.html', context)


def home3(request, cluster_id):
    global articles
    global sujets
    detaille = articles[articles.loc[:, 'id_cluster'] == cluster_id]
    json_records = detaille.reset_index().to_json(orient='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data, 'c': cluster_id, 'e': sujets}
    return render(request, 'html2.html', context)
