"""
Projet de dimensionnement d'un buffer en Python
Modélisation d'un nœud réseau avec chaîne de Markov à temps discret
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from IPython.display import Image, display


def afficher_image(filename, width=None):
    """Affiche une image dans un notebook Jupyter."""
    if os.path.exists(filename):
        if width:
            display(Image(filename, width=width))
        else:
            display(Image(filename))
        return True
    return False


def calculer_distribution_stationnaire(K, p0, p1, p2, p3=None):
    """
    Calcule la distribution de probabilité stationnaire π.
    Formule analytique : π[i] = π[0] * (p2/p0)^i pour i = 0 à K
    """
    prob_arrivee = p3 if (p3 is not None and p3 > 0) else p2
    somme = sum((prob_arrivee / p0) ** i for i in range(K + 1))
    pi = np.zeros(K + 1)
    pi[0] = 1.0 / somme
    for i in range(1, K + 1):
        pi[i] = pi[0] * (prob_arrivee / p0) ** i
    return pi


def nombre_moyen_clients(pi):
    """Calcule le nombre moyen de clients L = Σ(i=1 à K) i * π[i]"""
    K = len(pi) - 1
    return sum(i * pi[i] for i in range(1, K + 1))


def debit_entree(pi, K, p0, p1, p2, p3=None):
    """Calcule le débit d'entrée Xe (paquets acceptés par slot)"""
    if p3 is None or p3 == 0:
        Xe = 2 * p2 * sum(pi[i] for i in range(K)) + \
             1 * (p1 * sum(pi[i] for i in range(K + 1)) + p2 * pi[K])
    else:
        Xe = 0.0
        for i in range(K + 1):
            etat_apres_service = max(0, i - 1)
            Xe += pi[i] * p1 * 1
            if etat_apres_service <= K - 3:
                Xe += pi[i] * p3 * 3
            elif etat_apres_service == K - 2:
                Xe += pi[i] * p3 * 2
            elif etat_apres_service == K - 1:
                Xe += pi[i] * p3 * 1
    return Xe


def nombre_moyen_paquets_perdus(pi, K, p0, p1, p2, p3=None):
    """Calcule le nombre moyen de paquets perdus par slot"""
    if p3 is None or p3 == 0:
        return 1 * p2 * pi[K]
    else:
        Loss = 0.0
        for i in range(K + 1):
            etat_apres_service = max(0, i - 1)
            if etat_apres_service == K - 2:
                Loss += pi[i] * p3 * 1
            elif etat_apres_service == K - 1:
                Loss += pi[i] * p3 * 2
        return Loss


def temps_reponse_moyen(L, Xe):
    """Calcule le temps de réponse moyen R = L / Xe (formule de Little)"""
    return L / Xe if Xe > 0 else 0


def taux_utilisation(pi):
    """Calcule le taux d'utilisation U = 1 - π[0]"""
    return 1 - pi[0]


def analyser_buffer(K, p0, p1, p2, p3=None, verbose=True):
    """Effectue l'analyse complète du buffer pour une capacité K donnée"""
    somme_probas = p0 + p1 + p2 + (p3 if p3 is not None else 0)
    if abs(somme_probas - 1.0) > 1e-6:
        print(f"Attention : Les probabilites ne somment pas a 1 (somme = {somme_probas})")
        return None
    
    pi = calculer_distribution_stationnaire(K, p0, p1, p2, p3)
    L = nombre_moyen_clients(pi)
    Xe = debit_entree(pi, K, p0, p1, p2, p3)
    Loss = nombre_moyen_paquets_perdus(pi, K, p0, p1, p2, p3)
    R = temps_reponse_moyen(L, Xe)
    U = taux_utilisation(pi)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYSE POUR K = {K} (Questions Q2-Q7)")
        print(f"{'='*60}")
        print(f"\n[Q2] Distribution de probabilite stationnaire pi :")
        for i in range(len(pi)):
            print(f"  pi[{i}] = {pi[i]:.6f}")
        print(f"\n[Q3] Nombre moyen de clients dans le systeme L : {L:.6f} paquets")
        print(f"[Q4] Debit d'entree Xe : {Xe:.6f} paquets par slot")
        print(f"[Q5] Nombre moyen de paquets perdus par slot : {Loss:.6f} paquets")
        print(f"[Q6] Temps de reponse moyen R : {R:.6f} slots")
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


def creer_dossier_graphes():
    """Crée le dossier graphes/ s'il n'existe pas"""
    if not os.path.exists('graphes'):
        os.makedirs('graphes')


def graphe_distribution_stationnaire(resultats, config_name=""):
    """Génère le graphe de la distribution stationnaire π pour différentes valeurs de K"""
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
    
    for idx in range(len(resultats), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Q2 : Distribution de probabilité stationnaire π(K) {config_name}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    filename = f'graphes/Q2_distribution_stationnaire{config_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Graphe sauvegarde : {filename}")
    plt.close()


def graphe_metriques_en_fonction_K(resultats, config_name=""):
    """Génère les graphes des métriques en fonction de K"""
    creer_dossier_graphes()
    K_values = [res['K'] for res in resultats]
    L_values = [res['L'] for res in resultats]
    Xe_values = [res['Xe'] for res in resultats]
    Loss_values = [res['Loss'] for res in resultats]
    R_values = [res['R'] for res in resultats]
    U_values = [res['U']*100 for res in resultats]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(K_values, L_values, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_xlabel('Capacité K', fontsize=11)
    axes[0, 0].set_ylabel('L (paquets)', fontsize=11)
    axes[0, 0].set_title('Q3 : Nombre moyen de clients L', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(K_values, Xe_values, 's-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Capacité K', fontsize=11)
    axes[0, 1].set_ylabel('Xe (paquets/slot)', fontsize=11)
    axes[0, 1].set_title('Q4 : Débit d\'entrée Xe', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(K_values, Loss_values, '^-', linewidth=2, markersize=8, color='red')
    axes[0, 2].set_xlabel('Capacité K', fontsize=11)
    axes[0, 2].set_ylabel('Loss (paquets/slot)', fontsize=11)
    axes[0, 2].set_title('Q5 : Nombre moyen de paquets perdus', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    axes[1, 0].plot(K_values, R_values, 'd-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xlabel('Capacité K', fontsize=11)
    axes[1, 0].set_ylabel('R (slots)', fontsize=11)
    axes[1, 0].set_title('Q6 : Temps de réponse moyen R', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(K_values, U_values, '*-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].set_xlabel('Capacité K', fontsize=11)
    axes[1, 1].set_ylabel('U (%)', fontsize=11)
    axes[1, 1].set_title('Q7 : Taux d\'utilisation U', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 105])
    
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
    print(f"  Graphe sauvegarde : {filename}")
    plt.close()


def graphe_comparaison_configurations(resultats_orig, resultats_new):
    """Génère les graphes comparatifs entre les deux configurations"""
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
    
    axes[0, 0].plot(K_values, L_orig, 'o-', linewidth=2, markersize=8, 
                    label='Original (p0=0.4, p1=0.2, p2=0.4)', color='blue')
    axes[0, 0].plot(K_values, L_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau (p0=0.5, p1=0.2, p2=0.3)', color='red')
    axes[0, 0].set_xlabel('Capacité K', fontsize=11)
    axes[0, 0].set_ylabel('L (paquets)', fontsize=11)
    axes[0, 0].set_title('Q10 : Nombre moyen de clients L', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(K_values, Xe_orig, 'o-', linewidth=2, markersize=8,
                    label='Original', color='blue')
    axes[0, 1].plot(K_values, Xe_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau', color='red')
    axes[0, 1].set_xlabel('Capacité K', fontsize=11)
    axes[0, 1].set_ylabel('Xe (paquets/slot)', fontsize=11)
    axes[0, 1].set_title('Q10 : Débit d\'entrée Xe', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
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
    
    axes[1, 0].plot(K_values, R_orig, 'o-', linewidth=2, markersize=8,
                    label='Original', color='blue')
    axes[1, 0].plot(K_values, R_new, 's--', linewidth=2, markersize=8,
                    label='Nouveau', color='red')
    axes[1, 0].set_xlabel('Capacité K', fontsize=11)
    axes[1, 0].set_ylabel('R (slots)', fontsize=11)
    axes[1, 0].set_title('Q10 : Temps de réponse moyen R', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
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
    print(f"  Graphe sauvegarde : {filename}")
    plt.close()
