# Analyse de courbes de spectrométrie

### Familiarisation avec le dataset
Dans un premier temps j'ai voulu me familiariser avec le jeu de données, j'ai analysé comment il était structuré, et j'ai voulu représenter les données sous forme de graphe.
- Visualisation des données (visualisation des 100 premières courbes de chaque spectre) :
<img width="1536" height="752" alt="Data" src="https://github.com/user-attachments/assets/a13f9f52-b80e-46c4-ab8a-b70a55b3689a" />

J'ai remarqué que les courbes de Lin et COTON étaient très ressemblantes, donc je ferais plus attention à eux.
L'unité de mesure n'était pas clairement donnée, mais les données semblaient concorder.
- Statistiques par matière :
<img width="967" height="164" alt="image" src="https://github.com/user-attachments/assets/77a33a9c-d9dc-42db-8d11-577af99a1856" />

### Nettoyage des données

J'ai commencé à vérifier s'il y avait des données manquantes, des doublons... <br>
Pour "ACETATE", le nombre de groupes de doublons est de 19. Les autres n'en ont pas. <br>
J'ai alors fusionné les doublons, et j'ai ensuite voulu vérifier s'il y avait du bruit dans les données. <br>
J'ai pour cela considéré un spectre comme aberrant si sa moyenne ou son écart-type est trop éloigné de la distribution des autres spectres de la même matière. <br>
J'ai retiré les spectres jugés comme aberrants. <br>

J'ai pu remarquer que sur les 100 premiers spectres, il y avait une courbe de LIN qui était particulièrement éloignée, elle est retirée après le nettoyage des données :
- Avant nettoyage :

<img width="977" height="459" alt="image" src="https://github.com/user-attachments/assets/5f28fd88-36f0-4591-abfe-78bc0338f8dd" />

- Après nettoyage : 

<img width="940" height="453" alt="image" src="https://github.com/user-attachments/assets/3bc486e0-68eb-4c94-8b67-c3e6efdfec96" />

### Teste du modèle à utiliser

Après avoir nettoyé les données, j'ai voulu tester quel modèle de machine learning serait approprié pour mon algorithme.
J'ai pu faire des tests sur divers modèles de machine learning supervisés de classification qui semblent adaptés pour des données spectres :

- PLS-DA
- SVM (Machines à vecteurs de support)
- RF (Random Forest)
- 1D-CNN (Réseaux de neurones convolutifs 1D)

<br>

Pour ce faire j'ai évalué les modèles selon plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Ainsi qu'une matrice de confusion afin de mieux visualiser les bons et mauvais résultats.

- Dans un premier temps j'ai testé les modèles avec les données brutes pour voir si mon nettoyage du dataset allait améliorer les résultats par la suite.

Les matrices de confusion :

<img width="640" height="480" alt="MC_CNN1D" src="https://github.com/user-attachments/assets/c6152726-cd88-45cf-b950-2e780b599f5d" />

<img width="640" height="480" alt="MC_RF" src="https://github.com/user-attachments/assets/664d3512-5a54-4328-abe3-f6df3a83dd7b" />

<img width="640" height="480" alt="MC_SVM" src="https://github.com/user-attachments/assets/b2026e74-b1fc-4a53-8f7a-8c9b765c1c6a" />

<img width="640" height="480" alt="MC_PLSDA" src="https://github.com/user-attachments/assets/050d641b-0d95-452e-b75a-c83697004568" />

Métriques du modèle :

<img width="657" height="232" alt="Classement" src="https://github.com/user-attachments/assets/5b4f0cb0-c5fe-442b-b25d-c6c332bae9e9" />

On peut remarquer que le modèle 1D-CNN arrive en tête, mais sur la matrice de confusion, on peut clairement voir qu'il a quand même du mal à différencier le LIN et le COTON (les deux matériaux qui possèdent des courbes de spectre assez ressemblantes).

<br>

- Ensuite j'ai retesté mes modèles après le nettoyage des données.

<br>

Les matrices de confusion :

<img width="640" height="480" alt="MC_CNN1D" src="https://github.com/user-attachments/assets/70459b05-4e5e-4100-b515-59e89d16f6cc" />

<img width="640" height="480" alt="MC_RF" src="https://github.com/user-attachments/assets/aecf533d-1465-4b44-944e-6d728bb5aefa" />

<img width="640" height="480" alt="MC_SVM" src="https://github.com/user-attachments/assets/9b64004b-cf03-4380-8a7f-77448c2ad50f" />

<img width="640" height="480" alt="MC_PLSDA" src="https://github.com/user-attachments/assets/266a1d96-94a6-4a2b-a657-13dbe558fa2f" />

Métriques du modèle :

<img width="655" height="242" alt="Classement" src="https://github.com/user-attachments/assets/dd917ff3-23d4-4904-abe5-1460395acdd3" />

On peut remarquer que le modèle PLS-DA détient les meilleurs résultats, j'ai refait plusieurs tests:

<img width="646" height="235" alt="Classement2" src="https://github.com/user-attachments/assets/dcdd7017-c9af-4a33-ae70-58794efb2b41" />

<img width="640" height="226" alt="Classement3" src="https://github.com/user-attachments/assets/fcdd5dc7-396e-44bd-a368-59ef60910f6c" />

<img width="716" height="245" alt="Classement4" src="https://github.com/user-attachments/assets/41b95e63-fcaa-4383-92ef-ed21c152c9e5" />

Le modèle PLS-DA arrive toujours en tête, avec un score de 1 partout, j'ai donc décidé de l'utiliser pour faire mon algorithme.
Je pense aussi que ces résultats sont possibles car le jeu de données est assez précis, il n'y a pas beaucoup de bruit dans les données, mais aussi le volume n'est pas très conséquent.

### Algorithme
Enfin j'ai fait un algorithme qui utilise PLS-DA pour dans un premier temps nettoyer les données.
Ensuite, il entraîne le modèle et génère trois fichiers :

- pls_da.pkl → modèle PLS-DA entraîné
- scaler.pkl → standardiseur utilisé pour normaliser les spectres
- label_encoder.pkl → encodeur des classes (ACETATE, COTON, LIN, PES)

Ces fichiers sont utilisés afin de ne pas avoir à réentraîner le modèle à chaque appel du code.
L'utilisateur rentre le chemin vers le fichier contenant les données du spectre qu'on souhaite tester.
Et enfin, l'algorithme renvoie le résultat de la classification prédite.

<img width="431" height="236" alt="resultat_test" src="https://github.com/user-attachments/assets/5fe31d0b-ea4a-4ac2-8b65-30b62617a81f" />

J'ai pu réaliser plusieurs tests, et j'ai eu de bons résultats.

### Limites

- J'ai adapté le prétraitement à ce dataset, mais dans l'idéal et avec un dataset plus gros et plus bruyant, il faudrait mieux nettoyer les données, dans mon cas comme il n'y avait pas de données manquantes, je n'ai pas traité cet aspect, mais dans l'idéal il faudrait ajouter une fonction pour soit remplir ces données ou les supprimer.

- J'ai testé sur plusieurs modèles, mais il serait intéressant de lire quelques articles pour voir quel modèle pourrait être le plus adapté à ce type de dataset.

- Faire de la recherche pour trouver quelles sont les valeurs attendues pour un certain matériel et ainsi mieux définir scientifiquement quelles seraient les données dites aberrantes.
















