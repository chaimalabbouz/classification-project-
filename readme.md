# Classification d'émotions avec augmentation de données par VAE

Projet de classification de trois classes d'émotions (**NEGATIVE**, **NEUTRAL**, **POSITIVE**) à partir du dataset `emotions.csv`. Le pipeline combine réduction de dimension, génération de données synthétiques par **Variational Autoencoders (VAE)** et comparaison de plusieurs classifieurs.

## Pipeline

1. **Prétraitement**
   - Encodage des labels (`LabelEncoder`)
   - Standardisation des features (`StandardScaler`)
   - Suppression des outliers avec `IsolationForest` (contamination = 0.02)
   - Réduction de dimension par **PCA** (30 composantes) → jeu réel de 2089 échantillons

2. **Visualisation exploratoire**
   - **LDA** à 2 composantes pour visualiser la séparation des classes
   - **t-SNE** (perplexity = 40, learning rate = 300, 2000 itérations, init PCA) sur les données PCA

3. **Génération de données synthétiques (VAE)**
   - Un VAE entraîné **par classe** (NEGATIVE, POSITIVE, NEUTRAL) sur les features PCA
   - Architecture : encodeur `Linear(30→128)` → `μ`, `logσ²` (espace latent de dimension 10) → décodeur `Linear(10→128→30)`
   - Loss : MSE (reconstruction) + divergence KL
   - Optimiseur Adam (lr = 1e-3), batch size 32, 50 epochs, split 80/20
   - Génération de **400 échantillons par classe** (1200 au total)

   | Classe | Train loss finale | Test loss finale |
   |---|---|---|
   | NEGATIVE | 19.33 | 21.06 |
   | POSITIVE | 21.14 | 19.75 |
   | NEUTRAL | 3.18 | 4.51 |

4. **Jeu combiné**
   - Données réelles (PCA) + données générées = **3289 échantillons**
   - Répartition : NEUTRAL 1116, POSITIVE 1100, NEGATIVE 1073
   - Visualisation t-SNE des données générées

5. **Classification** (split 80/20 stratifié pour les modèles sklearn/XGBoost)

   | Modèle | Détails | Accuracy test |
   |---|---|---|
   | Random Forest | 200 arbres | 0.9438 |
   | XGBoost | 200 estimateurs, max_depth = 6, lr = 0.1 | 0.9696 |
   | MLP (PyTorch) | 30 → 128 → 128 → 3, ReLU | 0.9711 (epoch 50) |
   | MLP profond (PyTorch) | 30 → 256 → 128 → 64 → 3, ReLU, Dropout 0.3 | 0.9681 (epoch 50) |

   Les réseaux de neurones sont entraînés 50 epochs (Adam, lr = 1e-3, batch size 32, CrossEntropyLoss).

## Limites à garder en tête

- Le jeu de test des classifieurs est tiré du jeu **combiné** : il contient donc des échantillons synthétiques en plus des données réelles. Les accuracies reportées ne mesurent pas uniquement la performance sur données réelles.
- Les VAE sont entraînés sur les features déjà projetées par PCA (et non sur les features d'origine).

## Technologies

Python, pandas, NumPy, scikit-learn, PyTorch, XGBoost, Matplotlib

## Exécution

1. Placer `emotions.csv` (colonne `label` + features) à l'emplacement attendu par le notebook (chemin Kaggle : `/kaggle/input/vae-final-nchalla/emotions.csv`, à adapter si besoin).
2. Installer les dépendances :
   ```bash
   pip install pandas numpy matplotlib scikit-learn torch xgboost
   ```
3. Exécuter le notebook cellule par cellule.
