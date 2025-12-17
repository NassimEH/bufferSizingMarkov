# Projet de Dimensionnement d'un Buffer

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Ce projet implémente l'analyse d'un système de buffer dans un nœud réseau en utilisant une chaîne de Markov à temps discret. Il permet d'étudier le dimensionnement optimal d'un buffer en fonction de différentes capacités et distributions d'arrivées de paquets.

## Description

Le système modélise un nœud réseau (n1) qui reçoit des paquets et les transmet vers un autre nœud (n2). Le buffer a une capacité finie K et utilise un service déterministe (1 paquet par slot).

## Paramètres du système

- **Taille des paquets** : 64 octets
- **Débit du lien** : 512 Mbits/s
- **Unité de temps** : slot de 10⁻⁶ s (1 microseconde)
- **Service** : 1 paquet servi par slot
- **Capacité du buffer** : K (finie)

## Distribution d'arrivées

### Configuration originale
- p₀ = 0.4 (probabilité de 0 arrivée par slot)
- p₁ = 0.2 (probabilité de 1 arrivée par slot)
- p₂ = 0.4 (probabilité de 2 arrivées par slot)

### Configuration modifiée (Question 10)
- p₀ = 0.5 (probabilité de 0 arrivée par slot)
- p₁ = 0.2 (probabilité de 1 arrivée par slot)
- p₃ = 0.3 (probabilité de 3 arrivées par slot)

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python buffer_dimensionnement.py
```

Le programme effectue automatiquement :
1. Le calcul de la distribution stationnaire pour chaque valeur de K
2. Le calcul de toutes les métriques de performance
3. L'analyse pour K ∈ {2, 5, 10, 15, 80}
4. La comparaison entre les deux configurations de probabilités

## Métriques calculées

1. **Distribution stationnaire π** : Probabilités d'équilibre de chaque état
2. **L** : Nombre moyen de clients (paquets) dans le système
3. **Xe** : Débit d'entrée (nombre moyen de paquets acceptés par slot)
4. **Loss** : Nombre moyen de paquets perdus par slot
5. **R** : Temps de réponse moyen d'un paquet (formule de Little)
6. **U** : Taux d'utilisation du lien

## Structure du code

- `calculer_distribution_stationnaire()` : Calcule la distribution stationnaire π
- `nombre_moyen_clients()` : Calcule L
- `debit_entree()` : Calcule Xe
- `nombre_moyen_paquets_perdus()` : Calcule le taux de perte
- `temps_reponse_moyen()` : Calcule R (formule de Little)
- `taux_utilisation()` : Calcule U
- `analyser_buffer()` : Fonction principale d'analyse
- `main()` : Point d'entrée du programme

## Questions traitées

- **Q1** : Chaîne de Markov (graphe généré automatiquement pour K=5)
- **Q2** : Distribution stationnaire π(K)
- **Q3** : Nombre moyen de clients L
- **Q4** : Débit d'entrée Xe
- **Q5** : Nombre moyen de paquets perdus
- **Q6** : Temps de réponse moyen R
- **Q7** : Taux d'utilisation U
- **Q8** : Implémentation Python pour K ∈ {2, 5, 10, 15, 80}
- **Q9** : Commentaires sur les résultats
- **Q10** : Analyse avec probabilités modifiées

## Visualisations

Le programme génère automatiquement des graphes pour toutes les questions :

- **Q1** : Diagramme de la chaîne de Markov (graphe orienté)
- **Q2** : Histogrammes de la distribution stationnaire pour chaque K
- **Q3-Q7** : Graphiques des métriques en fonction de K
- **Q10** : Comparaisons entre les deux configurations

Tous les graphes sont sauvegardés dans le dossier `graphes/` au format PNG haute résolution (300 DPI).

## Structure du projet

```
.
├── buffer_dimensionnement.py    # Code principal
├── requirements.txt              # Dépendances Python
├── README.md                     # Documentation principale
├── .gitignore                    # Fichiers ignorés par Git
└── graphes/                      # Dossier contenant tous les graphes
    ├── README.md                 # Documentation des graphes
    ├── Q1_chaine_markov_K5.png
    ├── Q2_distribution_stationnaire(Original).png
    ├── Q2_distribution_stationnaire(Nouvelle).png
    ├── Q3-Q7_metriques_K(Original).png
    ├── Q3-Q7_metriques_K(Nouvelle).png
    └── Q10_comparaison_configurations.png
```

## Auteur

Projet réalisé dans le cadre de l'étude du dimensionnement d'un buffer en réseau. (Cours de File d'attente, TSP FISA 2A)

## Licence

Ce projet est sous licence MIT.

