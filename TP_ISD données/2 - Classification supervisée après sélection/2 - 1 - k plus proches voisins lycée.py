import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme() # permet d'obtenir le fonc gris avec les lignes blanches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold

#On reprend les variables sélectionnées précédemment et on tente de prédire le nombre d'élèves du même lycée par une méthode des k plus proches voisins.
Pred_lycee_df0 = pd.read_csv(r"2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_lycee", sep=";")
NOMS = Pred_lycee_df0['Indices'].to_list()
Pred_lycee_df = Pred_lycee_df0.drop(columns=['Unnamed: 0','Indices', 'acc_term_f', 'acc_aca_orig'])

#Affichage du nombre d'élèves de la même académie
sns.displot(Pred_lycee_df, x="acc_term", color="crimson")
plt.xlabel("Nombre d'élèves du même lycée")
plt.ylabel("Nombre de formations concernées")
plt.show()
#Observation : le nombre d'élèves venant du même lycée dépasse rarement 10, et est le plus souvent nul.

#Séparation de la variable à prédire et des autres
y0 = Pred_lycee_df["acc_term"]
y = y0.values
X0 = Pred_lycee_df.drop(columns="acc_term")
X = X0.values
VAR = X0.columns.to_list()

#Normalisation
scale= StandardScaler()
X_scale = scale.fit_transform(X)

#Méthode des k plus proches voisins par Train-Validation-Test Split
X_no_test, X_test, y_no_test, y_test = train_test_split(X_scale, y, test_size=0.15, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_no_test, y_no_test, test_size=0.20, random_state=7)

Scores = []

for k in range(1, 500):
  neigh = KNeighborsRegressor(n_neighbors=k)
  neigh.fit(X_train, y_train)
  Scores.append(neigh.score(X_val, y_val))

k_opt_val = np.argmax(Scores) + 1

print(f"Meilleur k validation par Train-Validation-Test Split : {k_opt_val}")

#Affichage de la performance du modèle selon le nombre de voisins (Train-Validation-Test Split)
plt.plot(range(1, 500), Scores, label="Erreur de validation (train-test split)")
plt.xlabel("Nombre de voisins k", fontsize=12)
plt.ylabel("R2", fontsize=12)
plt.title("Évaluation du meilleur hyperparamètre k pour le modèle", fontsize=12)
plt.legend()
plt.show()

#Affichage du meilleur score sur l'ensemble test
knn = KNeighborsRegressor(n_neighbors=k_opt_val)
knn.fit(X_no_test, y_no_test)
opt_score = knn.score(X_test, y_test)
print("Score (Train-Validation-Test Split) : " + str(opt_score))

#Méthode des k plus proches voisins par validation croisée
n_splits = 5
cv = KFold(n_splits=5)
Scores = []
for k in range(1, 500):
  neigh = KNeighborsRegressor(n_neighbors=k)
  neigh.fit(X_train, y_train)
  Scores.append(cross_val_score(neigh, X, y, cv=cv).mean())
k_opt_val2 = np.argmax(Scores) + 1
print(f"Meilleur k validation par validation croisée : {k_opt_val2}")

#Affichage de la performance du modèle selon le nombre de voisins
plt.plot(range(1, 500), Scores, label="Erreur de validation (Validation croisée)")
plt.xlabel("Nombre de voisins k", fontsize=12)
plt.ylabel("R2", fontsize=12)
plt.title("Évaluation du meilleur hyperparamètre k pour le modèle", fontsize=12)
plt.legend()
plt.show()

#Affichage du meilleur score sur l'ensemble test
knn = KNeighborsRegressor(n_neighbors=k_opt_val2)
knn.fit(X_no_test, y_no_test)
opt_score = knn.score(X_test, y_test)
print("Score (Validation croisée) : " + str(opt_score))

#Résultats :
#Pour le Train-Validation-Test-Split, le meilleur nombre de voisins est de 20.
#Le meilleur score R2 est 0.32352820439966057.
#Pour la Validation croisée (avec une séparation en 5 groupes), le meilleur nombre de voisins est de 39.
#Le meilleur score R2 est 0.32566173551482236.
#La méthode des k plus proches voisins ne permet pas d'obtenir de meilleurs scores que la forêt aléatoire.
