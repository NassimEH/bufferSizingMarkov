# Graphes générés - Projet de Dimensionnement d'un Buffer

Ce dossier contient tous les graphes générés automatiquement par le programme `buffer_dimensionnement.py`.

## Structure des fichiers

### Question 1 (Q1)
- **Q1_chaine_markov_K5.png** : Diagramme de la chaîne de Markov pour K=5
  - Représente les états (0 à 5) et les transitions possibles
  - Les arcs sont étiquetés avec les probabilités de transition

### Question 2 (Q2)
- **Q2_distribution_stationnaire(Original).png** : Distribution de probabilité stationnaire π pour différentes valeurs de K (configuration originale)
- **Q2_distribution_stationnaire(Nouvelle).png** : Distribution de probabilité stationnaire π pour différentes valeurs de K (configuration modifiée)
  - Affiche les probabilités π[i] pour chaque état i
  - Un sous-graphe par valeur de K

### Questions 3-7 (Q3-Q7)
- **Q3-Q7_metriques_K(Original).png** : Graphes des métriques en fonction de K (configuration originale)
- **Q3-Q7_metriques_K(Nouvelle).png** : Graphes des métriques en fonction de K (configuration modifiée)
  
  Contient 6 sous-graphes :
  - **Q3** : Nombre moyen de clients L (paquets)
  - **Q4** : Débit d'entrée Xe (paquets/slot)
  - **Q5** : Nombre moyen de paquets perdus Loss (paquets/slot) - échelle logarithmique
  - **Q6** : Temps de réponse moyen R (slots)
  - **Q7** : Taux d'utilisation U (%)
  - Vue d'ensemble normalisée

### Question 10 (Q10)
- **Q10_comparaison_configurations.png** : Comparaison entre les deux configurations
  - 6 sous-graphes comparant les métriques
  - Un sous-graphe supplémentaire montrant les ratios (Nouveau/Original)

## Configuration originale
- p₀ = 0.4 (probabilité de 0 arrivée)
- p₁ = 0.2 (probabilité de 1 arrivée)
- p₂ = 0.4 (probabilité de 2 arrivées)

## Configuration modifiée (Q10)
- p₀ = 0.5 (probabilité de 0 arrivée)
- p₁ = 0.2 (probabilité de 1 arrivée)
- p₂ = 0.0 (probabilité de 2 arrivées)
- p₃ = 0.3 (probabilité de 3 arrivées)

## Valeurs de K analysées
K ∈ {2, 5, 10, 15, 80}

## Format des fichiers
- Format : PNG
- Résolution : 300 DPI
- Taille : Optimisée pour l'affichage et l'impression

## Génération
Les graphes sont générés automatiquement lors de l'exécution de `buffer_dimensionnement.py`.
Ils sont sauvegardés dans ce dossier avec des noms explicites indiquant la question et la configuration.

