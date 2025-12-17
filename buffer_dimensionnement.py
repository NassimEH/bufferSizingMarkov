"""
Projet de dimensionnement d'un buffer en Python
Modélisation d'un nœud réseau avec chaîne de Markov à temps discret

QUESTIONS TRAITÉES :
- Q1 : Chaîne de Markov (représentation graphique pour K=5)
- Q2 : Distribution de probabilité stationnaire π(K)
- Q3 : Nombre moyen de clients dans le système L
- Q4 : Débit d'entrée (nombre moyen de paquets acceptés par slot) Xe
- Q5 : Nombre moyen de paquets perdus par slot Loss
- Q6 : Temps de réponse moyen d'un paquet R
- Q7 : Taux d'utilisation du lien U
- Q8 : Programme Python pour K ∈ {2, 5, 10, 15, 80}
- Q9 : Commentaires sur les résultats en fonction de K
- Q10 : Analyse avec probabilités modifiées (p0=0.5, p1=0.2, p3=0.3)
"""

import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os


# ============================================================================
# QUESTION 1 : CHAÎNE DE MARKOV
# ============================================================================
def afficher_chaine_markov(K, p0, p1, p2, p3=None):
    """
    QUESTION 1 : Affiche la description de la chaîne de Markov pour K donné.
    
    Pour K=5, la chaîne de Markov a 6 états (0 à 5) représentant le nombre
    de paquets dans le buffer.
    
    Pour chaque état i :
    - Service : 1 paquet servi si i > 0 → état devient max(0, i-1)
    - Arrivées : 
      * 0 arrivée (p0) : reste à l'état après service
      * 1 arrivée (p1) : passe à min(K, état_après_service + 1)
      * 2 arrivées (p2) : passe à min(K, état_après_service + 2)
      * 3 arrivées (p3) : passe à min(K, état_après_service + 3) [si applicable]
    
    Args:
        K: Capacité du buffer
        p0, p1, p2: Probabilités d'arrivée
        p3: Probabilité de 3 arrivées (optionnel)
    """
    print(f"\n{'='*70}")
    print(f"QUESTION 1 : CHAÎNE DE MARKOV POUR K = {K}")
    print(f"{'='*70}")
    print(f"\nLa chaîne de Markov a {K+1} états (0 à {K}) représentant le nombre")
    print(f"de paquets dans le buffer.\n")
    print(f"Probabilités d'arrivée :")
    print(f"  p0 = {p0} (0 arrivée)")
    print(f"  p1 = {p1} (1 arrivée)")
    print(f"  p2 = {p2} (2 arrivées)")
    if p3 is not None and p3 > 0:
        print(f"  p3 = {p3} (3 arrivées)")
    print(f"\nRègles de transition :")
    print(f"  1. Service : 1 paquet est servi par slot (si buffer non vide)")
    print(f"  2. Arrivées : 0, 1, 2 ou 3 paquets arrivent selon les probabilités")
    print(f"  3. Rejet : Les paquets en excès sont perdus si buffer plein (état {K})")
    print(f"\nExemple de transitions depuis l'état i :")
    print(f"  - État après service : max(0, i-1)")
    print(f"  - Avec 0 arrivée (p0) : reste à max(0, i-1)")
    print(f"  - Avec 1 arrivée (p1) : passe à min({K}, max(0, i-1) + 1)")
    print(f"  - Avec 2 arrivées (p2) : passe à min({K}, max(0, i-1) + 2)")
    if p3 is not None and p3 > 0:
        print(f"  - Avec 3 arrivées (p3) : passe à min({K}, max(0, i-1) + 3)")
    print(f"\nLa matrice de transition P[{K+1}x{K+1}] est construite automatiquement.")
    print(f"Pour une représentation graphique, voir le diagramme d'états.")


# ============================================================================
# QUESTION 2 : DISTRIBUTION DE PROBABILITÉ STATIONNAIRE π(K)
# ============================================================================
def calculer_distribution_stationnaire(K, p0, p1, p2, p3=None):
    """
    QUESTION 2 : Calcule la distribution de probabilité stationnaire π pour une capacité K.
    
    Formule : Résolution du système π = πP avec Σπ = 1
    où P est la matrice de transition de la chaîne de Markov.
    
    Pour chaque état i (0 à K), on calcule les probabilités d'équilibre.
    Les transitions dépendent des arrivées (0, 1, 2 ou 3 paquets) et du service (1 paquet/slot).
    
    Args:
        K: Capacité du buffer
        p0: Probabilité de 0 arrivée par slot
        p1: Probabilité de 1 arrivée par slot
        p2: Probabilité de 2 arrivées par slot
        p3: Probabilité de 3 arrivées par slot (optionnel, défaut=None)
    
    Returns:
        pi: Array de taille K+1 contenant les probabilités stationnaires π[i]
    """
    # Construction de la matrice de transition P
    # P[i][j] = probabilité de passer de l'état i à l'état j en un slot
    P = np.zeros((K+1, K+1))
    
    for i in range(K+1):
        # Service : 1 paquet est servi si i > 0
        etat_apres_service = max(0, i - 1)
        
        # Transitions selon les arrivées
        # 0 arrivée : reste à etat_apres_service
        P[i][etat_apres_service] += p0
        
        # 1 arrivée : passe à min(K, etat_apres_service + 1)
        nouvel_etat_1 = min(K, etat_apres_service + 1)
        P[i][nouvel_etat_1] += p1
        
        # 2 arrivées : passe à min(K, etat_apres_service + 2)
        nouvel_etat_2 = min(K, etat_apres_service + 2)
        P[i][nouvel_etat_2] += p2
        
        # 3 arrivées (si p3 est fourni)
        if p3 is not None and p3 > 0:
            nouvel_etat_3 = min(K, etat_apres_service + 3)
            P[i][nouvel_etat_3] += p3
    
    # Résolution du système π = πP avec Σπ = 1
    # Équivalent à (P^T - I)π = 0 avec Σπ = 1
    # On remplace la dernière équation par Σπ = 1
    A = (P.T - np.eye(K+1))
    A[-1, :] = 1  # Dernière ligne : Σπ = 1
    b = np.zeros(K+1)
    b[-1] = 1
    
    pi = solve(A, b)
    
    # Vérification que les probabilités sont positives
    pi = np.maximum(pi, 0)
    # Normalisation
    pi = pi / np.sum(pi)
    
    return pi


# ============================================================================
# QUESTION 3 : NOMBRE MOYEN DE CLIENTS DANS LE SYSTÈME L
# ============================================================================
def nombre_moyen_clients(pi):
    """
    QUESTION 3 : Calcule le nombre moyen de clients (paquets) dans le système.
    
    Formule : L = Σ(i=0 à K) i * π[i]
    
    Args:
        pi: Distribution stationnaire
    
    Returns:
        L: Nombre moyen de paquets dans le système
    """
    L = sum(i * pi[i] for i in range(len(pi)))
    return L


# ============================================================================
# QUESTION 4 : DÉBIT D'ENTRÉE (NOMBRE MOYEN DE PAQUETS ACCEPTÉS PAR SLOT) Xe
# ============================================================================
def debit_entree(pi, K, p0, p1, p2, p3=None):
    """
    QUESTION 4 : Calcule le débit d'entrée (nombre moyen de paquets acceptés par slot).
    
    Formule : Xe = Σ(i=0 à K) π[i] * (nombre de paquets acceptés depuis l'état i)
    
    Les paquets sont acceptés si le buffer n'est pas plein.
    - 0 arrivée : 0 paquet accepté
    - 1 arrivée : 1 paquet accepté si état < K, 0 sinon
    - 2 arrivées : 2 paquets acceptés si état < K-1, 1 si état = K-1, 0 si état = K
    - 3 arrivées : 3 paquets acceptés si état <= K-3, 2 si état = K-2, 1 si état = K-1, 0 si état = K
    
    Args:
        pi: Distribution stationnaire
        K: Capacité du buffer
        p0, p1, p2: Probabilités d'arrivée
        p3: Probabilité de 3 arrivées (optionnel)
    
    Returns:
        Xe: Débit d'entrée moyen (paquets par slot)
    """
    Xe = 0.0
    
    for i in range(K+1):
        # État après service
        etat_apres_service = max(0, i - 1)
        
        # 0 arrivée : 0 paquet accepté
        # (pas besoin d'ajouter)
        
        # 1 arrivée : 1 paquet accepté si etat_apres_service < K
        if etat_apres_service < K:
            Xe += pi[i] * p1 * 1
        
        # 2 arrivées : 
        # - 2 paquets acceptés si etat_apres_service <= K-2
        # - 1 paquet accepté si etat_apres_service = K-1
        # - 0 paquet accepté si etat_apres_service = K
        if etat_apres_service <= K - 2:
            Xe += pi[i] * p2 * 2
        elif etat_apres_service == K - 1:
            Xe += pi[i] * p2 * 1
        
        # 3 arrivées (si p3 est fourni)
        if p3 is not None and p3 > 0:
            if etat_apres_service <= K - 3:
                Xe += pi[i] * p3 * 3
            elif etat_apres_service == K - 2:
                Xe += pi[i] * p3 * 2
            elif etat_apres_service == K - 1:
                Xe += pi[i] * p3 * 1
    
    return Xe


# ============================================================================
# QUESTION 5 : NOMBRE MOYEN DE PAQUETS PERDUS PAR SLOT Loss
# ============================================================================
def nombre_moyen_paquets_perdus(pi, K, p0, p1, p2, p3=None):
    """
    QUESTION 5 : Calcule le nombre moyen de paquets perdus par slot.
    
    Formule : Loss = Σ(i=0 à K) π[i] * (nombre de paquets perdus depuis l'état i)
    
    Les paquets sont perdus lorsque le buffer est plein.
    - 1 arrivée : 1 paquet perdu si état = K (après service)
    - 2 arrivées : 1 ou 2 paquets perdus selon l'état
    - 3 arrivées : 1, 2 ou 3 paquets perdus selon l'état
    
    Args:
        pi: Distribution stationnaire
        K: Capacité du buffer
        p0, p1, p2: Probabilités d'arrivée
        p3: Probabilité de 3 arrivées (optionnel)
    
    Returns:
        Loss: Nombre moyen de paquets perdus par slot
    """
    Loss = 0.0
    
    for i in range(K+1):
        # État après service
        etat_apres_service = max(0, i - 1)
        
        # 1 arrivée : 1 paquet perdu si etat_apres_service = K
        if etat_apres_service == K:
            Loss += pi[i] * p1 * 1
        
        # 2 arrivées :
        # - 1 paquet perdu si etat_apres_service = K-1
        # - 2 paquets perdus si etat_apres_service = K
        if etat_apres_service == K - 1:
            Loss += pi[i] * p2 * 1
        elif etat_apres_service == K:
            Loss += pi[i] * p2 * 2
        
        # 3 arrivées (si p3 est fourni)
        if p3 is not None and p3 > 0:
            if etat_apres_service == K - 2:
                Loss += pi[i] * p3 * 1
            elif etat_apres_service == K - 1:
                Loss += pi[i] * p3 * 2
            elif etat_apres_service == K:
                Loss += pi[i] * p3 * 3
    
    return Loss


# ============================================================================
# QUESTION 6 : TEMPS DE RÉPONSE MOYEN D'UN PAQUET R
# ============================================================================
def temps_reponse_moyen(L, Xe):
    """
    QUESTION 6 : Calcule le temps de réponse moyen d'un paquet (formule de Little).
    
    Formule : R = L / Xe
    où L est le nombre moyen de clients et Xe est le débit d'entrée.
    
    Args:
        L: Nombre moyen de clients dans le système
        Xe: Débit d'entrée (paquets acceptés par slot)
    
    Returns:
        R: Temps de réponse moyen (en slots)
    """
    if Xe > 0:
        R = L / Xe
    else:
        R = 0
    return R


# ============================================================================
# QUESTION 7 : TAUX D'UTILISATION DU LIEN U
# ============================================================================
def taux_utilisation(pi):
    """
    QUESTION 7 : Calcule le taux d'utilisation du lien.
    
    Formule : U = 1 - π[0]
    Le lien est utilisé si le buffer n'est pas vide (état > 0).
    
    Args:
        pi: Distribution stationnaire
    
    Returns:
        U: Taux d'utilisation (proportion du temps où le lien est occupé)
    """
    U = 1 - pi[0]
    return U


def analyser_buffer(K, p0, p1, p2, p3=None, verbose=True):
    """
    Effectue l'analyse complète du buffer pour une capacité K donnée.
    Implémente les questions Q2 à Q7.
    
    Args:
        K: Capacité du buffer
        p0, p1, p2: Probabilités d'arrivée
        p3: Probabilité de 3 arrivées (optionnel)
        verbose: Si True, affiche les résultats
    
    Returns:
        dict: Dictionnaire contenant tous les résultats
    """
    # Vérification que les probabilités somment à 1
    somme_probas = p0 + p1 + p2 + (p3 if p3 is not None else 0)
    if abs(somme_probas - 1.0) > 1e-6:
        print(f"Attention : Les probabilités ne somment pas à 1 (somme = {somme_probas})")
        return None
    
    # Q2 : Distribution stationnaire
    pi = calculer_distribution_stationnaire(K, p0, p1, p2, p3)
    
    # Q3 : Nombre moyen de clients dans le système
    L = nombre_moyen_clients(pi)
    
    # Q4 : Débit d'entrée
    Xe = debit_entree(pi, K, p0, p1, p2, p3)
    
    # Q5 : Nombre moyen de paquets perdus
    Loss = nombre_moyen_paquets_perdus(pi, K, p0, p1, p2, p3)
    
    # Q6 : Temps de réponse moyen
    R = temps_reponse_moyen(L, Xe)
    
    # Q7 : Taux d'utilisation
    U = taux_utilisation(pi)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYSE POUR K = {K} (Questions Q2-Q7)")
        print(f"{'='*60}")
        print(f"\n[Q2] Distribution de probabilité stationnaire π :")
        for i in range(len(pi)):
            print(f"  π[{i}] = {pi[i]:.6f}")
        
        print(f"\n[Q3] Nombre moyen de clients dans le système L : {L:.6f} paquets")
        print(f"[Q4] Débit d'entrée Xe : {Xe:.6f} paquets par slot")
        print(f"[Q5] Nombre moyen de paquets perdus par slot : {Loss:.6f} paquets")
        print(f"[Q6] Temps de réponse moyen R : {R:.6f} slots")
        print(f"[Q7] Taux d'utilisation du lien U : {U:.6f} ({U*100:.2f}%)")
    
    return {
        'K': K,
        'pi': pi,
        'L': L,
        'Xe': Xe,
        'Loss': Loss,
        'R': R,
        'U': U
    }


# ============================================================================
# FONCTIONS DE VISUALISATION - GÉNÉRATION DE GRAPHES
# ============================================================================

def creer_dossier_graphes():
    """Crée le dossier graphes/ s'il n'existe pas."""
    if not os.path.exists('graphes'):
        os.makedirs('graphes')
        print("Dossier 'graphes' créé.")


def graphe_chaine_markov(K, p0, p1, p2, p3=None):
    """
    QUESTION 1 : Génère le graphe de la chaîne de Markov pour K donné.
    
    Args:
        K: Capacité du buffer
        p0, p1, p2: Probabilités d'arrivée
        p3: Probabilité de 3 arrivées (optionnel)
    """
    creer_dossier_graphes()
    
    G = nx.DiGraph()
    
    # Ajouter les nœuds (états)
    for i in range(K+1):
        G.add_node(i)
    
    # Ajouter les arcs (transitions)
    # Utiliser un dictionnaire pour accumuler les probabilités
    transitions = {}
    
    for i in range(K+1):
        etat_apres_service = max(0, i - 1)
        
        # 0 arrivée
        if p0 > 0:
            cle = (i, etat_apres_service)
            if cle in transitions:
                transitions[cle] += p0
            else:
                transitions[cle] = p0
        
        # 1 arrivée
        if p1 > 0:
            nouvel_etat = min(K, etat_apres_service + 1)
            cle = (i, nouvel_etat)
            if cle in transitions:
                transitions[cle] += p1
            else:
                transitions[cle] = p1
        
        # 2 arrivées
        if p2 > 0:
            nouvel_etat = min(K, etat_apres_service + 2)
            cle = (i, nouvel_etat)
            if cle in transitions:
                transitions[cle] += p2
            else:
                transitions[cle] = p2
        
        # 3 arrivées
        if p3 is not None and p3 > 0:
            nouvel_etat = min(K, etat_apres_service + 3)
            cle = (i, nouvel_etat)
            if cle in transitions:
                transitions[cle] += p3
            else:
                transitions[cle] = p3
    
    # Ajouter les arcs au graphe
    for (i, j), proba in transitions.items():
        G.add_edge(i, j, weight=proba, label=f'{proba:.2f}')
    
    # Positionnement des nœuds
    pos = {}
    if K <= 10:
        # Disposition linéaire pour petits K
        for i in range(K+1):
            pos[i] = (i * 2, 0)
    else:
        # Disposition circulaire pour grands K
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Création de la figure
    plt.figure(figsize=(14, 8))
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                           node_size=1500, alpha=0.9)
    
    # Dessiner les arcs
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2,
                           alpha=0.6, edge_color='gray', arrows=True,
                           arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # Dessiner les labels des nœuds
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Dessiner les labels des arcs (probabilités)
    edge_labels = {(u, v): G[u][v]['label'] for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title(f'Q1 : Chaîne de Markov pour K = {K}\n'
              f'p0={p0}, p1={p1}, p2={p2}' + 
              (f', p3={p3}' if p3 is not None and p3 > 0 else ''),
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    filename = f'graphes/Q1_chaine_markov_K{K}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Graphe sauvegardé : {filename}")
    plt.close()


def graphe_distribution_stationnaire(resultats, config_name=""):
    """
    QUESTION 2 : Génère le graphe de la distribution stationnaire π pour différentes valeurs de K.
    
    Args:
        resultats: Liste de dictionnaires contenant les résultats
        config_name: Nom de la configuration (pour le titre)
    """
    creer_dossier_graphes()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, res in enumerate(resultats):
        if idx >= len(axes):
            break
        
        K = res['K']
        pi = res['pi']
        etats = list(range(len(pi)))
        
        axes[idx].bar(etats, pi, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('État i (nombre de paquets)', fontsize=10)
        axes[idx].set_ylabel('Probabilité π[i]', fontsize=10)
        axes[idx].set_title(f'K = {K}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks(etats)
    
    # Masquer les axes inutilisés
    for idx in range(len(resultats), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Q2 : Distribution de probabilité stationnaire π(K) {config_name}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'graphes/Q2_distribution_stationnaire{config_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Graphe sauvegardé : {filename}")
    plt.close()


def graphe_metriques_en_fonction_K(resultats, config_name=""):
    """
    QUESTIONS 3-7 : Génère les graphes des métriques en fonction de K.
    
    Args:
        resultats: Liste de dictionnaires contenant les résultats
        config_name: Nom de la configuration (pour le titre)
    """
    creer_dossier_graphes()
    
    K_values = [res['K'] for res in resultats]
    L_values = [res['L'] for res in resultats]
    Xe_values = [res['Xe'] for res in resultats]
    Loss_values = [res['Loss'] for res in resultats]
    R_values = [res['R'] for res in resultats]
    U_values = [res['U']*100 for res in resultats]  # En pourcentage
    
    # Créer une figure avec plusieurs sous-graphes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Q3 : Nombre moyen de clients L
    axes[0, 0].plot(K_values, L_values, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_xlabel('Capacité K', fontsize=11)
    axes[0, 0].set_ylabel('L (paquets)', fontsize=11)
    axes[0, 0].set_title('Q3 : Nombre moyen de clients L', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q4 : Débit d'entrée Xe
    axes[0, 1].plot(K_values, Xe_values, 's-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Capacité K', fontsize=11)
    axes[0, 1].set_ylabel('Xe (paquets/slot)', fontsize=11)
    axes[0, 1].set_title('Q4 : Débit d\'entrée Xe', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q5 : Taux de perte Loss
    axes[0, 2].plot(K_values, Loss_values, '^-', linewidth=2, markersize=8, color='red')
    axes[0, 2].set_xlabel('Capacité K', fontsize=11)
    axes[0, 2].set_ylabel('Loss (paquets/slot)', fontsize=11)
    axes[0, 2].set_title('Q5 : Nombre moyen de paquets perdus', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')  # Échelle logarithmique pour mieux voir les petites valeurs
    
    # Q6 : Temps de réponse moyen R
    axes[1, 0].plot(K_values, R_values, 'd-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xlabel('Capacité K', fontsize=11)
    axes[1, 0].set_ylabel('R (slots)', fontsize=11)
    axes[1, 0].set_title('Q6 : Temps de réponse moyen R', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q7 : Taux d'utilisation U
    axes[1, 1].plot(K_values, U_values, '*-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].set_xlabel('Capacité K', fontsize=11)
    axes[1, 1].set_ylabel('U (%)', fontsize=11)
    axes[1, 1].set_title('Q7 : Taux d\'utilisation U', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 105])
    
    # Vue d'ensemble : toutes les métriques normalisées
    axes[1, 2].plot(K_values, np.array(L_values)/max(L_values), 'o-', label='L (norm)', linewidth=2)
    axes[1, 2].plot(K_values, np.array(Xe_values)/max(Xe_values), 's-', label='Xe (norm)', linewidth=2)
    axes[1, 2].plot(K_values, np.array(U_values)/100, '*-', label='U (norm)', linewidth=2)
    axes[1, 2].set_xlabel('Capacité K', fontsize=11)
    axes[1, 2].set_ylabel('Valeur normalisée', fontsize=11)
    axes[1, 2].set_title('Vue d\'ensemble (normalisée)', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Questions Q3-Q7 : Métriques en fonction de K {config_name}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'graphes/Q3-Q7_metriques_K{config_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Graphe sauvegardé : {filename}")
    plt.close()


def graphe_comparaison_configurations(resultats_orig, resultats_new):
    """
    QUESTION 10 : Génère les graphes comparatifs entre les deux configurations.
    
    Args:
        resultats_orig: Résultats de la configuration originale
        resultats_new: Résultats de la nouvelle configuration
    """
    creer_dossier_graphes()
    
    K_values = [res['K'] for res in resultats_orig]
    L_orig = [res['L'] for res in resultats_orig]
    Xe_orig = [res['Xe'] for res in resultats_orig]
    Loss_orig = [res['Loss'] for res in resultats_orig]
    R_orig = [res['R'] for res in resultats_orig]
    U_orig = [res['U']*100 for res in resultats_orig]
    
    L_new = [res['L'] for res in resultats_new]
    Xe_new = [res['Xe'] for res in resultats_new]
    Loss_new = [res['Loss'] for res in resultats_new]
    R_new = [res['R'] for res in resultats_new]
    U_new = [res['U']*100 for res in resultats_new]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # L : Nombre moyen de clients
    axes[0, 0].plot(K_values, L_orig, 'o-', linewidth=2, markersize=8, 
                    label='Original (p0=0.4, p1=0.2, p2=0.4)', color='blue')
    axes[0, 0].plot(K_values, L_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau (p0=0.5, p1=0.2, p3=0.3)', color='red')
    axes[0, 0].set_xlabel('Capacité K', fontsize=11)
    axes[0, 0].set_ylabel('L (paquets)', fontsize=11)
    axes[0, 0].set_title('Q10 : Nombre moyen de clients L', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Xe : Débit d'entrée
    axes[0, 1].plot(K_values, Xe_orig, 'o-', linewidth=2, markersize=8,
                    label='Original', color='blue')
    axes[0, 1].plot(K_values, Xe_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau', color='red')
    axes[0, 1].set_xlabel('Capacité K', fontsize=11)
    axes[0, 1].set_ylabel('Xe (paquets/slot)', fontsize=11)
    axes[0, 1].set_title('Q10 : Débit d\'entrée Xe', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss : Taux de perte
    axes[0, 2].plot(K_values, Loss_orig, 'o-', linewidth=2, markersize=8,
                    label='Original', color='blue')
    axes[0, 2].plot(K_values, Loss_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau', color='red')
    axes[0, 2].set_xlabel('Capacité K', fontsize=11)
    axes[0, 2].set_ylabel('Loss (paquets/slot)', fontsize=11)
    axes[0, 2].set_title('Q10 : Taux de perte Loss', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # R : Temps de réponse
    axes[1, 0].plot(K_values, R_orig, 'o-', linewidth=2, markersize=8,
                    label='Original', color='blue')
    axes[1, 0].plot(K_values, R_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau', color='red')
    axes[1, 0].set_xlabel('Capacité K', fontsize=11)
    axes[1, 0].set_ylabel('R (slots)', fontsize=11)
    axes[1, 0].set_title('Q10 : Temps de réponse moyen R', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # U : Taux d'utilisation
    axes[1, 1].plot(K_values, U_orig, 'o-', linewidth=2, markersize=8,
                    label='Original', color='blue')
    axes[1, 1].plot(K_values, U_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau', color='red')
    axes[1, 1].set_xlabel('Capacité K', fontsize=11)
    axes[1, 1].set_ylabel('U (%)', fontsize=11)
    axes[1, 1].set_title('Q10 : Taux d\'utilisation U', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 105])
    
    # Ratio nouveau/original
    ratio_L = [new/orig if orig > 0 else 0 for new, orig in zip(L_new, L_orig)]
    ratio_Xe = [new/orig if orig > 0 else 0 for new, orig in zip(Xe_new, Xe_orig)]
    ratio_U = [new/orig if orig > 0 else 0 for new, orig in zip(U_new, U_orig)]
    
    axes[1, 2].plot(K_values, ratio_L, 'o-', label='L (ratio)', linewidth=2)
    axes[1, 2].plot(K_values, ratio_Xe, 's-', label='Xe (ratio)', linewidth=2)
    axes[1, 2].plot(K_values, ratio_U, '*-', label='U (ratio)', linewidth=2)
    axes[1, 2].axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 2].set_xlabel('Capacité K', fontsize=11)
    axes[1, 2].set_ylabel('Ratio (Nouveau/Original)', fontsize=11)
    axes[1, 2].set_title('Q10 : Ratio Nouveau/Original', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Q10 : Comparaison Configuration Originale vs Nouvelle',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = 'graphes/Q10_comparaison_configurations.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Graphe sauvegardé : {filename}")
    plt.close()


def generer_tous_graphes(resultats, resultats_new=None, p0=0.4, p1=0.2, p2=0.4, p3=None):
    """
    Génère tous les graphes pour toutes les questions.
    
    Args:
        resultats: Résultats de la configuration originale
        resultats_new: Résultats de la nouvelle configuration (optionnel)
        p0, p1, p2, p3: Probabilités d'arrivée de la configuration originale
    """
    print(f"\n{'='*70}")
    print("GÉNÉRATION DES GRAPHES")
    print(f"{'='*70}")
    
    # Q1 : Chaîne de Markov pour K=5 (configuration originale)
    print("\n[Q1] Génération du graphe de la chaîne de Markov (K=5)...")
    graphe_chaine_markov(K=5, p0=p0, p1=p1, p2=p2, p3=p3)
    
    # Q2 : Distribution stationnaire
    print("\n[Q2] Génération du graphe de la distribution stationnaire...")
    graphe_distribution_stationnaire(resultats, config_name="(Original)")
    
    # Q3-Q7 : Métriques en fonction de K
    print("\n[Q3-Q7] Génération des graphes des métriques...")
    graphe_metriques_en_fonction_K(resultats, config_name="(Original)")
    
    # Q10 : Comparaison si nouvelle configuration fournie
    if resultats_new is not None:
        print("\n[Q10] Génération des graphes de comparaison...")
        graphe_distribution_stationnaire(resultats_new, config_name="(Nouvelle)")
        graphe_metriques_en_fonction_K(resultats_new, config_name="(Nouvelle)")
        graphe_comparaison_configurations(resultats, resultats_new)
    
    print(f"\n{'='*70}")
    print("✓ Tous les graphes ont été générés et sauvegardés dans le dossier 'graphes/'")
    print(f"{'='*70}\n")


def main():
    """
    Fonction principale : implémente toutes les questions du projet.
    Q1 : Chaîne de Markov (affichage pour K=5)
    Q8 : Programme pour différentes valeurs de K (2, 5, 10, 15, 80)
    Q9 : Commentaires sur les résultats en fonction de K
    Q10 : Analyse avec probabilités modifiées
    """
    print("="*70)
    print("PROJET DE DIMENSIONNEMENT D'UN BUFFER")
    print("="*70)
    
    # Paramètres initiaux
    p0, p1, p2 = 0.4, 0.2, 0.4
    valeurs_K = [2, 5, 10, 15, 80]
    
    # ========================================================================
    # QUESTION 1 : CHAÎNE DE MARKOV POUR K=5
    # ========================================================================
    afficher_chaine_markov(K=5, p0=p0, p1=p1, p2=p2)
    
    # ========================================================================
    # QUESTION 8 : PROGRAMME POUR DIFFÉRENTES VALEURS DE K
    # ========================================================================
    print(f"\n{'='*70}")
    print("QUESTION 8 : ANALYSE POUR DIFFÉRENTES VALEURS DE K")
    print(f"{'='*70}")
    
    print(f"\nParamètres d'arrivée :")
    print(f"  p0 = {p0} (probabilité de 0 arrivée)")
    print(f"  p1 = {p1} (probabilité de 1 arrivée)")
    print(f"  p2 = {p2} (probabilité de 2 arrivées)")
    print(f"\nValeurs de K à analyser : {valeurs_K}")
    
    # Stockage des résultats
    resultats = []
    
    # Analyse pour chaque valeur de K (Q8)
    print(f"\n{'='*70}")
    print("CALCULS POUR CHAQUE VALEUR DE K (Questions Q2-Q7)")
    print(f"{'='*70}")
    for K in valeurs_K:
        resultat = analyser_buffer(K, p0, p1, p2, verbose=True)
        if resultat:
            resultats.append(resultat)
    
    # Tableau récapitulatif (Q8)
    print(f"\n{'='*70}")
    print("QUESTION 8 : TABLEAU RÉCAPITULATIF POUR TOUTES LES VALEURS DE K")
    print(f"{'='*70}")
    print(f"{'K':<5} {'L (moy)':<12} {'Xe':<12} {'Loss':<12} {'R (slots)':<12} {'U (%)':<10}")
    print("-" * 60)
    for res in resultats:
        print(f"{res['K']:<5} {res['L']:<12.6f} {res['Xe']:<12.6f} {res['Loss']:<12.6f} "
              f"{res['R']:<12.6f} {res['U']*100:<10.2f}")
    
    # ========================================================================
    # QUESTION 9 : COMMENTAIRES SUR LES RÉSULTATS EN FONCTION DE K
    # ========================================================================
    print(f"\n{'='*70}")
    print("QUESTION 9 : COMMENTAIRES SUR LES RÉSULTATS EN FONCTION DE K")
    print(f"{'='*70}")
    print("\nAnalyse de l'impact de K sur les performances :")
    print("-" * 70)
    
    if len(resultats) >= 2:
        print(f"\n1. Nombre moyen de clients (L) :")
        for res in resultats:
            print(f"   K={res['K']:2d} : L={res['L']:.4f} paquets")
        
        print(f"\n2. Débit d'entrée (Xe) :")
        for res in resultats:
            print(f"   K={res['K']:2d} : Xe={res['Xe']:.4f} paquets/slot")
        
        print(f"\n3. Taux de perte (Loss) :")
        for res in resultats:
            print(f"   K={res['K']:2d} : Loss={res['Loss']:.6f} paquets/slot "
                  f"({res['Loss']/res['Xe']*100 if res['Xe']>0 else 0:.2f}% du débit d'entrée)")
        
        print(f"\n4. Temps de réponse moyen (R) :")
        for res in resultats:
            print(f"   K={res['K']:2d} : R={res['R']:.4f} slots")
        
        print(f"\n5. Taux d'utilisation (U) :")
        for res in resultats:
            print(f"   K={res['K']:2d} : U={res['U']*100:.2f}%")
    
    # ========================================================================
    # QUESTION 10 : MODIFICATION DES PROBABILITÉS D'ARRIVÉE
    # ========================================================================
    print(f"\n{'='*70}")
    print("QUESTION 10 : MODIFICATION DES PROBABILITÉS D'ARRIVÉE")
    print(f"{'='*70}")
    print("\nNouvelles probabilités d'arrivée :")
    print(f"  p0 = 0.5 (probabilité de 0 arrivée)")
    print(f"  p1 = 0.2 (probabilité de 1 arrivée)")
    print(f"  p3 = 0.3 (probabilité de 3 arrivées)")
    print("\nNote : Le système original avait p2=0.4 (2 arrivées).")
    print("      Le nouveau système a p3=0.3 (3 arrivées) et p2=0.")
    
    # Nouvelles probabilités
    p0_new, p1_new, p2_new, p3_new = 0.5, 0.2, 0.0, 0.3
    
    print(f"\nAnalyse avec les nouvelles probabilités (Questions Q2-Q7) :")
    print("-" * 70)
    
    # Analyse avec les nouvelles probabilités (p0=0.5, p1=0.2, p2=0.0, p3=0.3)
    resultats_new = []
    print(f"\n{'='*70}")
    print("CALCULS POUR CHAQUE VALEUR DE K AVEC NOUVELLES PROBABILITÉS")
    print(f"{'='*70}")
    for K in valeurs_K:
        resultat = analyser_buffer(K, p0_new, p1_new, p2_new, p3_new, verbose=True)
        if resultat:
            resultats_new.append(resultat)
    
    # Tableau récapitulatif pour les nouvelles probabilités (Q10)
    print(f"\n{'='*70}")
    print("QUESTION 10 : TABLEAU RÉCAPITULATIF (NOUVELLES PROBABILITÉS)")
    print(f"{'='*70}")
    print(f"{'K':<5} {'L (moy)':<12} {'Xe':<12} {'Loss':<12} {'R (slots)':<12} {'U (%)':<10}")
    print("-" * 60)
    for res in resultats_new:
        print(f"{res['K']:<5} {res['L']:<12.6f} {res['Xe']:<12.6f} {res['Loss']:<12.6f} "
              f"{res['R']:<12.6f} {res['U']*100:<10.2f}")
    
    # Comparaison entre les deux configurations (Q10)
    print(f"\n{'='*70}")
    print("QUESTION 10 : COMPARAISON CONFIGURATION ORIGINALE vs NOUVELLE")
    print(f"{'='*70}")
    print(f"\nConfiguration originale : p0=0.4, p1=0.2, p2=0.4")
    print(f"Configuration nouvelle   : p0=0.5, p1=0.2, p3=0.3")
    print(f"\n{'K':<5} {'L (orig)':<12} {'L (new)':<12} {'Xe (orig)':<12} {'Xe (new)':<12} "
          f"{'Loss (orig)':<12} {'Loss (new)':<12}")
    print("-" * 80)
    for i, K in enumerate(valeurs_K):
        if i < len(resultats) and i < len(resultats_new):
            print(f"{K:<5} {resultats[i]['L']:<12.6f} {resultats_new[i]['L']:<12.6f} "
                  f"{resultats[i]['Xe']:<12.6f} {resultats_new[i]['Xe']:<12.6f} "
                  f"{resultats[i]['Loss']:<12.6f} {resultats_new[i]['Loss']:<12.6f}")
    
    print(f"\nQUESTION 10 : Commentaires sur les résultats avec les nouvelles probabilités :")
    print("-" * 70)
    print("Avec p3=0.3 (3 arrivées possibles), le système subit des pics d'arrivée")
    print("plus importants, ce qui peut augmenter :")
    print("- Le nombre moyen de clients dans le système (L)")
    print("- Le taux de perte (Loss) pour les petites valeurs de K")
    print("- Le temps de réponse moyen (R)")
    print("\nCependant, avec p0=0.5 (plus de slots sans arrivée), cela peut")
    print("compenser partiellement l'effet des pics d'arrivée.")
    
    # ========================================================================
    # GÉNÉRATION DE TOUS LES GRAPHES
    # ========================================================================
    generer_tous_graphes(resultats, resultats_new, p0=p0, p1=p1, p2=p2, p3=None)
    # Note : p3=None pour la configuration originale, p3_new sera utilisé dans la fonction


if __name__ == "__main__":
    main()

