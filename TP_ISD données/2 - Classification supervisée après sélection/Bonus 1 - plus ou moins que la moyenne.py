#Ici, on va tenter de prédire si le nombre d'élèves venant de la même académie est supérieur ou inférieur à la moyenne de ce nombre (déterminée sur l'échantillon ici étudié)
#On utilisera d'abord les k plus proches voisins puis un RandomForest, en représentant les résultats avec des matrices de confusion.

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
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Fonction pour afficher la matrice de confusion
def model_evaluation(classifier, X_testing, y_testing):

  # CONFUSION MATRIX
  cm = confusion_matrix(y_testing, classifier.predict(X_testing))
  names = ['True Neg','False Pos','False Neg','True Pos']
  counts = [value for value in cm.flatten()]
  percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')

  tn,fp,fn,tp = cm.flatten()

  # ACCURACY
  print('ACCURACY : ','{0:.2%}'.format((tp + tn)/(tp + fp + tn + fn)))

  # PRECISION
  print('PRECISION : ','{0:.2%}'.format(tp/(tp + fp)))

  # RECALL
  print('RECALL : ','{0:.2%}'.format(tp/(tp + fn)))


Pred_acad_df0 = pd.read_csv(r"2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad", sep=";")
NOMS = Pred_acad_df0['Indices'].to_list()
Pred_acad_df = Pred_acad_df0.drop(columns=['Unnamed: 0','Indices', 'acc_term', 'acc_term_f'])
print((Pred_acad_df["acc_aca_orig"]).mean()) #On affiche le nombre moyen d'élèves provenant de la même académie
#On trouve un résultat d'environ 15.695, que l'on arrondit à 16.
#On va alors classer les formations en deux groupes : celles pour qui le nombre d'élèves de la même académie est strictement inférieur à 16, et celles pour qui cette valeur est supérieure à 16.
Pred_acad_a = Pred_acad_df.copy()
Pred_acad_a["meme_acc_sup16"] = (Pred_acad_df["acc_aca_orig"] >= 16).astype('category')
Pred_acad_b = Pred_acad_a.drop(columns="acc_aca_orig")

#On affiche le nombre de formations avec moins ou plus de 16 élèves de la même académie
sns.displot(Pred_acad_b, x="meme_acc_sup16", color="crimson")
plt.xlabel("Nombre d'élèves supérieur ou inférieur à 16")
plt.ylabel("Nombre de formations")
plt.xticks([0, 1])
plt.show()

#Séparation de la variable à prédire et des features
y0 = Pred_acad_b["meme_acc_sup16"]
y = y0.values
X0 = Pred_acad_b.drop(columns="meme_acc_sup16")
X = X0.values
var = X0.columns.to_list()

#Normalisation
scale= StandardScaler()
X_scale = scale.fit_transform(X)

X_no_test, X_test, y_no_test, y_test = train_test_split(X_scale, y, test_size=0.15, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_no_test, y_no_test, test_size=0.20, random_state=7)

#Méthode des k plus proches voisins (Train-test-split)
Scores = []

for k in range(1, 500):
  neigh = KNeighborsClassifier(n_neighbors=k)
  neigh.fit(X_train, y_train)
  Scores.append(neigh.score(X_val, y_val))

k_opt_val = np.argmax(Scores) + 1

print(f"Meilleur k validation  : {k_opt_val}")

#Affichage de l'erreur selon le nombre de voisins (Train-test split)
plt.plot(range(1, 500), Scores, label="Erreur de validation (train-test split)")
plt.xlabel("Nombre de voisins k", fontsize=12)
plt.ylabel("Scores", fontsize=12)
plt.title("Évaluation du meilleur hyperparamètre k pour le modèle", fontsize=12)
plt.legend()
plt.show()

#Matrice de confusion pour k plus proches voisins (train-test-split)
neigh = KNeighborsClassifier(n_neighbors=k_opt_val)
neigh.fit(X_train, y_train)
model_evaluation(neigh, X_test, y_test)



#Méthode des k plus proches voisins (Validation croisée)
n_splits = 5
cv = KFold(n_splits=5)

Scores2 = []

for k in range(1, 500):
  neigh = KNeighborsClassifier(n_neighbors=k)
  neigh.fit(X_train, y_train)
  Scores2.append(cross_val_score(neigh, X, y, cv=cv).mean())

k_opt_val2 = np.argmax(Scores) + 1

print(f"Meilleur k validation : {k_opt_val2}")

#Affichage du score selon le nombre de voisins
plt.plot(range(1, 500), Scores2, label="Erreur de validation (train-test split)")
plt.xlabel("Nombre de voisins k", fontsize=12)
plt.ylabel("Scores", fontsize=12)
plt.title("Évaluation du meilleur hyperparamètre k pour le modèle", fontsize=12)
plt.legend()
plt.show()

#Matrice de confusion pour k plus proches voisins (validation croisée)
neigh2 = KNeighborsClassifier(n_neighbors=k_opt_val2)
neigh2.fit(X_no_test, y_no_test)
model_evaluation(neigh2, X_test, y_test)



#Méthode de Forêt Aléatoire
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 20],
}
rf_model = RandomForestClassifier()
grid_search = GridSearchCV(rf_model, param_grid, cv=5).fit(X_no_test, y_no_test)
best_params_forest = grid_search.best_params_
best_score_forest = grid_search.best_score_
print("Meilleurs paramètres : " + str(best_params_forest))

#Matrice de confusion pour Forêt Aléatoire
rf_model = RandomForestClassifier(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
model_evaluation(rf_model, X_test, y_test)

#Résultats
#Pour l'apprentissage supervisé par k plus proches voisins (par train-test-split), le meilleur nombre de voisins est 11.
#Outre la matrice de confusion disponible dans les images, on a les performances suivantes :
#Exactitude :  85.79%
#Précision :  87.73%
#Rappel :  76.38%

#En déterminant le nombre de voisins par validation croisée, ce dernier reste fixé à 11. On obtient une matrice un peu différente, avec les scores suivants :
#Exactitude :  84.58%
#Précision :  85.93%
#Rappel :  75.08%

#Enfin, en passant par une forêt aléatoire, on trouve pour meilleurs paramètres {'max_depth': 20, 'n_estimators': 100}.
#On obtient de plus des scores un peu plus élevés (peu importe celui considéré) : 
#Exactitude :  89.95%
#Précision :  89.26%
#Rappel :  86.08%

#Au final, ce modèle en lui-même fonctionne assez bien. 
#Cette approche de classification peut être intéressante à considérer si l'on souhaite se pencher davantage sur la comparaison entre lycées, plutôt que sur les nombres bruts.
#Il serait possible d'effectuer des modèles similaires en prenant la médiane au lieu de la moyenne, ou encore le premier et le troisième quartiles. 
