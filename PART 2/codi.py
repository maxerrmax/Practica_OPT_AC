"""
Autòmat Cel·lular Elemental de Wolfram (1D)
============================================
Implementa les 256 regles de Wolfram per a autòmats cel·lulars elementals.
També inclou una versió de "gra gruixut" (coarse-grained) amb K=2.

Condició frontera: periòdica (els extrems es connecten entre si).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. FUNCIONS BASE
# ─────────────────────────────────────────────

def get_rule_table(rule_number: int) -> dict:
    """
    Genera la taula de transició per a una regla de Wolfram donada (0-255).
    Retorna un diccionari {(esquerra, centre, dreta): nou_estat}.
    """
    if not 0 <= rule_number <= 255:
        raise ValueError("La regla ha d'estar entre 0 i 255.")
    
    bits = format(rule_number, '08b')  # 8 bits, p.ex. "00011110" per regla 30
    rule_table = {}
    
    # Els 8 patrons possibles (111, 110, ..., 000) mapejats als bits de la regla
    for i, pattern in enumerate(range(7, -1, -1)):
        left   = (pattern >> 2) & 1
        center = (pattern >> 1) & 1
        right  = pattern & 1
        rule_table[(left, center, right)] = int(bits[i])
    
    return rule_table


def get_neighborhood(state: np.ndarray, i: int) -> tuple:
    """
    Retorna el veïnat (esquerra, centre, dreta) de la cel·la i.
    Condició frontera: PERIÒDICA (toroïdal).
    """
    n = len(state)
    return state[(i - 1) % n], state[i], state[(i + 1) % n]


def evolve_step(state: np.ndarray, rule_table: dict) -> np.ndarray:
    """
    Calcula la generació següent d'un estat donada una taula de regles.
    Utilitza get_neighborhood per obtenir el veïnat de cada cel·la.
    """
    n = len(state)
    new_state = np.zeros(n, dtype=int)

    for i in range(n):
        neighborhood = get_neighborhood(state, i)
        new_state[i] = rule_table[neighborhood]

    return new_state


def run_automaton(rule_number: int, width: int = 101, generations: int = 50,
                  initial_state: np.ndarray = None) -> np.ndarray:
    """
    Executa l'autòmat cel·lular elemental de Wolfram.

    Paràmetres:
        rule_number   : regla de Wolfram (0-255)
        width         : nombre de cel·les
        generations   : nombre de generacions a simular
        initial_state : estat inicial (si None, una sola cel·la central encesa)

    Retorna:
        Matriu 2D [generations+1 x width] amb l'evolució completa.
    """
    rule_table = get_rule_table(rule_number)
    
    # Estat inicial per defecte: una sola cel·la al centre
    if initial_state is None:
        state = np.zeros(width, dtype=int)
        state[width // 2] = 1
    else:
        state = initial_state.copy()
    
    history = [state.copy()]
    
    for _ in range(generations):
        state = evolve_step(state, rule_table)
        history.append(state.copy())
    
    return np.array(history)


# ─────────────────────────────────────────────
# 2. GRA GRUIXUT (COARSE-GRAINING)
# ─────────────────────────────────────────────

def coarse_grain(history: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Aplica un gra gruixut de factor K a posteriori sobre l'evolució temporal
    d'un CA original prèviament simulat amb run_automaton.
    
    Cada bloc de K cel·les contigües s'agrupa matemàticament en una de sola:
      - Valor del bloc = 1 si la majoria de les K cel·les valen 1, 0 altrament.

    Retorna una matriu de dimensions [generations x (width//k)].
    """
    gens, width = history.shape
    new_width = width // k
    coarse = np.zeros((gens, new_width), dtype=int)
    
    for g in range(gens):
        for i in range(new_width):
            bloc = history[g, i*k : i*k + k]
            coarse[g, i] = 1 if np.sum(bloc) >= k / 2 else 0
    
    return coarse


# ─────────────────────────────────────────────
# 3. REGLA FORMAL DEL CA DE GRA GRUIXUT
# ─────────────────────────────────────────────

def majority(bloc: list) -> int:
    """Regla de majoria: retorna 1 si la meitat o més de les cel·les valen 1."""
    return 1 if sum(bloc) >= len(bloc) / 2 else 0


def build_coarse_rule(rule_number: int, k: int = 2) -> dict:
    """
    Construeix la taula de transició formal del CA de gra gruixut.

    Segueix el següent procediment:
      Per cada combinació de 3 supracel·les (s_left, s_center, s_right) ∈ {0,1}^3:
        1. Expandeix cada supracel·la al seu bloc canònic: 0→[0,0,...], 1→[1,1,...]
        2. Aplica la regla original 1 pas sobre el context complet de 3k cel·les,
           amb padding fix als extrems (s_left i s_right) per simular continuïtat.
        3. Projecta el bloc central resultant a {0,1} amb la regla de majoria.

    Retorna un diccionari {(s_esq, s_cen, s_dre): nou_estat_supracel·la}.
    """
    rule_table = get_rule_table(rule_number)

    coarse_rule = {}

    for s_left in range(2):
        for s_center in range(2):
            for s_right in range(2):

                # Traduïm cada estat de supracel·la {0,1} a un bloc físic de K cel·les elementals iguals.
                bl = [s_left]   * k
                bc = [s_center] * k
                br = [s_right]  * k

                # Construïm el context complet d'avaluació. Afegim 1 cel·la extra de "padding" 
                # a cada extrem per permetre calcular correctament l'evolució de les vores dels blocs.
                padded = np.array([s_left] + bl + bc + br + [s_right], dtype=int)
                
                # Identifiquem els índexs del nostre bloc central dins l'array 'padded'.
                # Comença un cop passats el padding (1) i el bloc complet esquerre (k).
                c_start = 1 + k
                c_end   = 1 + 2 * k

                # Apliquem un únic pas de la regla elemental original sobre tot el context disponible.
                new_padded = padded.copy()
                for i in range(1, len(padded) - 1):
                    new_padded[i] = rule_table[(padded[i-1], padded[i], padded[i+1])]

                # Bloc central resultant i projecció per majoria
                new_center = list(new_padded[c_start:c_end])
                coarse_rule[(s_left, s_center, s_right)] = majority(new_center)

    return coarse_rule


def run_coarse_automaton(rule_number: int, width: int = 100, generations: int = 50,
                         k: int = 2) -> np.ndarray:
    """
    Simula completament de zero l'autòmat cel·lular de gra gruixut.

    En lloc d'aplicar el filtre sobre un resultat prèviament simulat,
    parteix d'un estat inicial reduït (una única supracel·la central encesa) i 
    evoluciona automàticament pas a pas usant exclusivament la taula de transició 
    formal construïda per build_coarse_rule.
    """
    coarse_rule = build_coarse_rule(rule_number, k)
    n_super = width // k

    state = np.zeros(n_super, dtype=int)
    state[n_super // 2] = 1

    history = [state.copy()]

    for _ in range(generations):
        new_state = np.zeros(n_super, dtype=int)
        for i in range(n_super):
            l = state[(i - 1) % n_super]
            c = state[i]
            r = state[(i + 1) % n_super]
            new_state[i] = coarse_rule[(l, c, r)]
        state = new_state
        history.append(state.copy())

    return np.array(history)


# ─────────────────────────────────────────────
# 4. VISUALITZACIÓ
# ─────────────────────────────────────────────

def plot_single_rule(rule_number: int, width: int = 101, generations: int = 50):
    """Visualitza l'evolució d'una sola regla."""
    history = run_automaton(rule_number, width, generations)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(history, cmap='binary', interpolation='nearest', aspect='auto')
    ax.set_title(f'Regla {rule_number} de Wolfram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cel·la')
    ax.set_ylabel('Generació')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'rule_{rule_number}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[✓] Gràfic guardat: rule_{rule_number}.png")


def plot_coarse_comparison(rule_number: int, width: int = 100, generations: int = 50, k: int = 2):
    """
    Compara l'evolució original amb la versió de gra gruixut (posteriori)
    i la versió obtinguda directament per la regla formal.
    """
    # 1. Original
    history = run_automaton(rule_number, width, generations)
    # 2. Gra gruixut a posteriori
    coarse  = coarse_grain(history, k)
    # 3. Gra gruixut amb regla formal
    formal_coarse = run_coarse_automaton(rule_number, width, generations, k)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Regla {rule_number}: Original vs Gruixut Posterior vs Formal (K={k})', 
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(history, cmap='binary', interpolation='nearest', aspect='auto')
    axes[0].set_title(f'Original ({width} cel·les)')
    axes[0].set_xlabel('Cel·la')
    axes[0].set_ylabel('Generació')
    
    axes[1].imshow(coarse, cmap='binary', interpolation='nearest', aspect='auto')
    axes[1].set_title(f'Gra Gruixut a posterior ({width//k} cel·les)')
    axes[1].set_xlabel('Supracel·la')
    axes[1].set_ylabel('Generació')
    
    axes[2].imshow(formal_coarse, cmap='binary', interpolation='nearest', aspect='auto')
    axes[2].set_title(f'Regla Formal directa ({width//k} cel·les)')
    axes[2].set_xlabel('Supracel·la')
    axes[2].set_ylabel('Generació')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'rule_{rule_number}_coarse_full_k{k}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[✓] Comparació completa guardada: rule_{rule_number}_coarse_full_k{k}.png")


def plot_multiple_rules(rules: list, width: int = 101, generations: int = 40):
    """
    Mostra diverses regles en una graella per comparar-les.
    """
    n = len(rules)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    
    for idx, rule_number in enumerate(rules):
        history = run_automaton(rule_number, width, generations)
        axes[idx].imshow(history, cmap='binary', interpolation='nearest', aspect='auto')
        axes[idx].set_title(f'Regla {rule_number}', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Amaguem els subplots buits
    for idx in range(len(rules), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Comparació de Regles de Wolfram', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multiple_rules.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[✓] Comparació múltiple guardada: multiple_rules.png")


def plot_combined_rules(rule_list: list, width: int = 101, generations: int = 50):
    """
    Combina diverses regles: aplica cada regla alternant generació a generació.
    Exemple: generació 0->1 amb regla A, generació 1->2 amb regla B, etc.
    """
    rule_tables = [get_rule_table(r) for r in rule_list]
    
    state = np.zeros(width, dtype=int)
    state[width // 2] = 1
    history = [state.copy()]
    
    for gen in range(generations):
        rt = rule_tables[gen % len(rule_list)]
        state = evolve_step(state, rt)
        history.append(state.copy())
    
    history = np.array(history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(history, cmap='binary', interpolation='nearest', aspect='auto')
    label = '+'.join(str(r) for r in rule_list)
    ax.set_title(f'Combinació de regles [{label}] (alternant)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cel·la')
    ax.set_ylabel('Generació')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'combined_rules_{label}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[✓] Combinació guardada: combined_rules_{label}.png")



# ─────────────────────────────────────────────
# 5. EXECUCIÓ PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == '__main__':
    
    # A continuació es proposa un exemple de programa per veure com funciona el codi.
    # Es pot modificar segons el que ens interessi visualitzar.

    print("=" * 60)
    print("  AUTÒMAT CEL·LULAR ELEMENTAL DE WOLFRAM")
    print("=" * 60)
    
    # --- A) Regla famosa: 30 (caòtica, usada com a generador pseudoaleatori)
    print("\n[1] Visualitzant Regla 30 (caòtica)...")
    plot_single_rule(rule_number=30, width=101, generations=50)
    
    # --- B) Regla 110 (Turing-completa)
    print("\n[2] Visualitzant Regla 110 (Turing-completa)...")
    plot_single_rule(rule_number=110, width=101, generations=50)
    
    # --- C) Regla 90 (Triangle de Sierpinski)
    print("\n[3] Visualitzant Regla 90 (Sierpinski)...")
    plot_single_rule(rule_number=90, width=101, generations=50)
    
    # --- D) Comparació original vs gra gruixut K=2 (Regla 30)
    print("\n[4] Comparació Regla 30: original vs gra gruixut K=2...")
    plot_coarse_comparison(rule_number=30, width=100, generations=50, k=2)
    
    # --- E) Comparació original vs gra gruixut K=2 (Regla 90)
    print("\n[5] Comparació Regla 90: original vs gra gruixut K=2...")
    plot_coarse_comparison(rule_number=90, width=100, generations=50, k=2)
    
    # --- F) Múltiples regles destacades
    print("\n[6] Mostrant regles interessants en graella...")
    interesting_rules = [30, 54, 60, 62, 90, 94, 102, 110, 122, 126, 150, 182]
    plot_multiple_rules(interesting_rules, width=101, generations=40)
    
    # --- G) Combinació de regles (alternant 30 i 90)
    print("\n[7] Combinació de regles 30 i 90 (alternant)...")
    plot_combined_rules([30, 90], width=101, generations=50)

    print("\n[✓] Totes les figures han estat generades correctament.")
    print("    Fitxers de sortida:")
    print("    - rule_30.png")
    print("    - rule_110.png")
    print("    - rule_90.png")
    print("    - rule_30_coarse_full_k2.png")
    print("    - rule_90_coarse_full_k2.png")
    print("    - multiple_rules.png")
    print("    - combined_rules_30+90.png")