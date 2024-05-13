from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Modèle similaire au RandomForest avec sélection finale, en cherchant à prédire le pourcentage d'admis venant de la même académie
#Ici, il s'agit juste de voir si les variables significatives changent beaucoup si l'on passe des effectifs au pourcentage.
def Modele(X, y, test_size, param_grid) :
    X_no_test, X_test, y_no_test, y_test = train_test_split(X, y, test_size=test_size, random_state=3)
    rf_model = RandomForestRegressor()
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="r2").fit(X_no_test, y_no_test)
    best_params_forest = grid_search.best_params_
    best_score_forest = grid_search.best_score_
    print("Meilleur paramètre : " + str(best_params_forest))
    print("Meilleur score évaluation : " + str(best_score_forest))

    #Appliquer le modèle aux données test
    rf_model = RandomForestRegressor(**best_params_forest)
    rf_model.fit(X_no_test, y_no_test)
    best_score_forest = rf_model.score(X_test, y_test)
    print("Meilleur score : " + str(best_score_forest))
    return(X_no_test, y_no_test, X_test, y_test, best_params_forest)

def Permutation(X_no_test, y_no_test, rf_model, index) :
    #Vérifier l'importance de chaque variable
    result = permutation_importance(
    rf_model, X_no_test, y_no_test, n_repeats=10, random_state=2, scoring='r2')
    importance_var_lycee_complet = pd.Series(result.importances_mean, index=index)
    fig, ax = plt.subplots()
    importance_var_lycee_complet.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Importance des variables déterminée par permutation")
    ax.set_ylabel("Diminution moyenne du score R2")
    fig.tight_layout()
    plt.show()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(importance_var_lycee_complet)
        
formations = pd.read_csv(r'2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad', sep=";")
formations.pop("Indices")
y_admis_acad = formations["acc_aca_orig"]/formations["capa_fin"] #Effectif des admis de la même académie selon la capacité de la formation
formations = formations.drop(["acc_term", "acc_term_f", "acc_aca_orig"], axis=1)

scaler = StandardScaler()
X_formations = scaler.fit_transform(formations)


param_grid = {
    'n_estimators': [150, 100],
    'max_depth': [None, 20],
}
index = formations.columns.to_list()

X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele(X_formations, y_admis_acad, 0.15, param_grid)
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Permutation(X_no_test, y_no_test, rf_model, index)

#En regardant le graphique de test par permutation, on remarque que : 
#- Toutes les variables concernant les effectifs ne se retrouvent pas réduites en importance avec le passage en pourcentage. Par exemple, acc_pp (nombre d'admis en phase principale) reste toujours une variable importante.
#- D'autres variables gagnent un peu plus en importance, notamment des variables de nature géographique : acc_mies_Paris (appartenance à l'académie de Paris), latitude et longitude.
