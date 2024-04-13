"""
Nom: TIGA
Prenom: Abdoul-Wakilou
Filière: Génie Logiciel - IFRI de l'Université d'Abomey-Calavi
Email: abdoulwakiloutiga@gmail.com
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title('Clustering kmeans des clients du centre commercial')

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    return df

df = load_data()

# Sélectionner les caractéristiques
features = st.multiselect('Sélectionnez les caractéristiques pour le clustering :',
                          df.columns.tolist(), default=['Age', 'Annual Income (k$)'])

# Créer un dataframe avec les caractéristiques sélectionnées
dfa = df[features]

# Standardiser les données
scaler = StandardScaler()
dfa_std = scaler.fit_transform(dfa)

# Appliquer KMeans
num_clusters = st.slider('Choisissez le nombre de clusters :', min_value=2, max_value=10, value=3)
kmeans = KMeans(n_clusters=num_clusters).fit(dfa_std)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

new_dfa = pd.DataFrame(data=dfa_std, columns=features)
new_dfa['labels_kmeans'] = labels

# Affichage de la visualisation avec les centroids
fig, ax = plt.subplots(figsize=[10, 7])

for i in range(num_clusters):
    cluster_data = new_dfa[new_dfa['labels_kmeans'] == i]
    ax.scatter(cluster_data.iloc[:, 1], cluster_data.iloc[:, 0], s=100, label=f'Cluster {i+1}')
    ax.scatter(centroids[i][1], centroids[i][0], s=200, marker='*', c=[cluster_data['labels_kmeans'].unique()], label='Centroide')
    ax.text(centroids[i][1], centroids[i][0], f'Centroide {i+1}', ha='center', va='bottom', fontsize=12)

ax.set_xlabel(features[1])
ax.set_ylabel(features[0])
ax.legend()
st.pyplot(fig)
