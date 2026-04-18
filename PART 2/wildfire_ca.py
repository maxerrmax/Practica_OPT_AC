"""
Autòmat Cel·lular per a la Propagació d'Incendis Forestals
===========================================================
Model m:n-CA^k (Multi-n-Dimensional Cellular Automaton).

Notació formal:
  - m : nombre de capes (layers)     → 5 capes base (+1 capa opcional de vent)
  - n : dimensió de cada capa        → 2 (espai bidimensional ℤ²)
  - k : nombre de capes principals   → 1 (capa d'estat del foc)
  Representació: 5:2-CA^1 sobre ℤ²

Capes del model (E_m):
  E1[x1,x2] : Humitat     — hores de protecció restants (decreix amb el temps)
  E2[x1,x2] : Vegetació   — hores de combustible restants (decreix quan crema)
  E3[x1,x2] : Estat foc   — UNBURNED=0, BURNING=1, BURNED=2 (Capa principal, k=1)
  E4[x1,x2] : Relleu      — elevació (afecta i accelera/frena la propagació)
  E5[x1,x2] : Combustible — força de propagació del foc local
  (Extra)   : Vent        — capa de biaix/vector obtinguda a partir de polígons

Funció d'evolució Λ (només definida a la capa principal E3):
  - Veïnatge: funció de veïnatge de Moore (8-connexitat):
      vn(x1,x2) = {(x1±1, x2±1), (x1±1, x2), (x1, x2±1)}
  - Nucli: nc(x1,x2) = {(x1,x2)}  (una sola cel·la)
  - Condició frontera: fixa (UNBURNED als extrems)

Format de dades: IDRISI32 (Raster: .doc + .img | Vectorial: .dvc + .vec)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import sys

# Reutilització de la Part 1 (Wolfram): regla de majoria.
# Si no es pot importar (execució fora de l'estructura esperada), fem fallback.
PART1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PART1_DIR not in sys.path:
    sys.path.append(PART1_DIR)

try:
    import codi as wolfram
except Exception:
    wolfram = None


if wolfram is not None:
    wolfram_majority = wolfram.majority
    wolfram_coarse_grain = wolfram.coarse_grain
else:
    # Fallback local si no es pot importar codi.py
    def wolfram_majority(bloc: list) -> int:
        return 1 if sum(bloc) >= len(bloc) / 2 else 0

    def wolfram_coarse_grain(history: np.ndarray, k: int = 2) -> np.ndarray:
        gens, width = history.shape
        new_width = width // k
        coarse = np.zeros((gens, new_width), dtype=int)
        for g in range(gens):
            for i in range(new_width):
                bloc = history[g, i*k : i*k + k]
                coarse[g, i] = 1 if np.sum(bloc) >= k / 2 else 0
        return coarse

# ─────────────────────────────────────────────
# ESTATS DEL CA  (capa principal E3)
# ─────────────────────────────────────────────
UNBURNED = 0   # Pendent de cremar-se
BURNING  = 1   # Cremant-se  (emet foc als veïns)
BURNED   = 2   # Ja cremat


# ═══════════════════════════════════════════════════════════════════════
#  LECTURA DE FITXERS IDRISI32 - CAPES TIPUS RASTER (.doc / .img)
# ═══════════════════════════════════════════════════════════════════════

def read_idrisi32_doc(filepath: str) -> dict:
    """
    Llegeix el fitxer de metadades IDRISI32 (.doc).
    Retorna un diccionari amb els paràmetres del raster.
    """
    meta = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                meta[key.strip().lower()] = val.strip()
    return meta


def read_idrisi32_img(filepath: str, rows: int, cols: int) -> np.ndarray:
    """
    Llegeix el fitxer de dades IDRISI32 (.img) en format ASCII.
    Retorna una matriu numpy de forma (rows, cols).
    """
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            values.extend([float(v) for v in line.split()])
    arr = np.array(values[:rows * cols], dtype=float).reshape(rows, cols)
    return arr


def load_raster_layer(doc_path: str, img_path: str) -> tuple[np.ndarray, dict]:
    """
    Carrega una capa raster IDRISI32 composta de fitxer .doc + .img.
    Retorna (matriu_dades, metadades).
    """
    meta = read_idrisi32_doc(doc_path)
    rows = int(meta.get('rows', 0))
    cols = int(meta.get('columns', 0))
    data = read_idrisi32_img(img_path, rows, cols)
    print(f"[✓] Capa carregada: '{meta.get('file title', '?')}' "
          f"({rows}x{cols}), rang [{data.min():.1f}, {data.max():.1f}]")
    return data, meta


# ═══════════════════════════════════════════════════════════════════════
#  LECTURA DE DADES VECTORIALS IDRISI32 (.vec / .dvc)  — opcional: vent
# ═══════════════════════════════════════════════════════════════════════

def read_idrisi32_vec(dvc_path: str, vec_path: str) -> list[list[tuple]]:
    """
    Llegeix fitxers vectorials IDRISI32 (.dvc + .vec).
    Retorna una llista de polígons, cada un com a llista de (x, y).

    Format .vec:
      id  n_punts
      x1 y1
      x2 y2
      ...
      0 0   (marcador de fi de polígon)
    """
    polygons = []
    with open(vec_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2:
            _pid, _n = int(parts[0]), int(parts[1])
            i += 1
            pts = []
            while i < len(lines):
                xy = lines[i].split()
                x, y = float(xy[0]), float(xy[1])
                if x == 0.0 and y == 0.0:
                    i += 1
                    break
                pts.append((x, y))
                i += 1
            if pts:
                polygons.append(pts)
        else:
            i += 1
    print(f"[✓] Capa vectorial carregada: {len(polygons)} polígon(s)")
    return polygons


def polygon_to_wind_bias(polygons: list, rows: int, cols: int,
                          x_min: float, x_max: float,
                          y_min: float, y_max: float) -> np.ndarray:
    """
    Converteix polígons vectorials a una matriu de biaix de vent [0..1].
    Les cel·les dins d'un polígon reben un biaix. Aquest biaix es
    calcula en funció de la fracció de polígons que les contenen (pot ser 
    >0 fins a 1.0).

    El biaix s'utilitza per ampliar la probabilitat de propagació
    en la direcció preferent del vent.
    """
    from matplotlib.path import Path

    bias = np.zeros((rows, cols), dtype=float)

    # Coordenades de cada cel·la al sistema de referència del vector
    cell_xs = np.linspace(x_min, x_max, cols)
    cell_ys = np.linspace(y_min, y_max, rows)
    xx, yy = np.meshgrid(cell_xs, cell_ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    for poly_pts in polygons:
        path = Path(poly_pts)
        inside = path.contains_points(points).reshape(rows, cols)
        bias += inside.astype(float)

    if bias.max() > 0:
        bias /= bias.max()   # Normalitzem a [0,1]

    return bias


def estimate_wind_vector_from_polygons(polygons: list[list[tuple[float, float]]]) -> np.ndarray:
    """
    Estima una direcció global de vent (vector unitari) a partir dels polígons.

    Idea: calculem l'eix principal (PCA 2D) del conjunt de punts i l'usem com
    a direcció preferent de propagació.
    """
    if not polygons:
        return np.array([1.0, 0.0], dtype=float)

    pts = []
    for poly in polygons:
        pts.extend(poly)

    arr = np.array(pts, dtype=float)
    if arr.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=float)

    centered = arr - arr.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vec = eigvecs[:, int(np.argmax(eigvals))]

    norm = np.linalg.norm(principal_vec)
    if norm == 0:
        return np.array([1.0, 0.0], dtype=float)

    unit = principal_vec / norm
    return np.array([float(unit[0]), float(unit[1])], dtype=float)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL m:n-CA^k  —  AUTÒMAT CEL·LULAR DE PROPAGACIÓ D'INCENDI
# ═══════════════════════════════════════════════════════════════════════

class WildfireCA:
    """
    Model 5:2-CA^1 per a la propagació d'incendis forestals.

    Capes (E_m sobre ℤ²):
      humidity   (E1) : temps de protecció restant per a cada cel·la
      vegetation (E2) : combustible restant (hores de crema)
      fire_state (E3) : estat del foc — capa principal (k=1)
      relief     (E4) : elevació (accelera/frena propagació segons pendent)
      fuel       (E5) : combustible local (força de propagació)
      wind_bias  (Ex) : biaix direccional de vent (capa opcional)

    Funció d'evolució Λ (definida per a E3):
      Veïnatge Moore 8-connexitat amb nucli = {(x1,x2)}.
      Funció de combinació Ψ: per a cada veí cremant, es modula la propagació 
      per factors de vent, relleu i combustible. Quan la propagació acumulada
      supera la humitat local de la cel·la, s'encén.
    """

    MOORE_OFFSETS = [(-1,-1),(-1,0),(-1,1),
                     ( 0,-1),        ( 0,1),
                     ( 1,-1),( 1,0),( 1,1)]

    def __init__(self,
                 humidity:    np.ndarray,
                 vegetation:  np.ndarray,
                 relief:      np.ndarray | None = None,
                 fuel:        np.ndarray | None = None,
                 wind_bias:   np.ndarray | None = None,
                 wind_vector: np.ndarray | None = None):
        self.rows, self.cols = humidity.shape
        assert vegetation.shape == humidity.shape, "Dimensions incompatibles (vegetació)"
        if relief is not None:
            assert relief.shape == humidity.shape, "Dimensions incompatibles (relleu)"
        if fuel is not None:
            assert fuel.shape == humidity.shape, "Dimensions incompatibles (combustible)"

        # ── Capes del model ──────────────────────────────────────────
        # E1: humitat residual per cel·la (decreix amb el temps)
        self.humidity   = humidity.copy().astype(float)
        # E2: vegetació/combustible restant (decreix mentre crema)
        self.vegetation = vegetation.copy().astype(float)
        # E4: relleu/elevació (afecta la propagació)
        self.relief = relief.copy().astype(float) if relief is not None else np.zeros_like(humidity)
        # E5: combustible local (força de propagació)
        self.fuel = fuel.copy().astype(float) if fuel is not None else np.ones_like(humidity)
        # E3: estat del foc (capa principal k=1)
        self.fire_state = np.full((self.rows, self.cols), UNBURNED, dtype=int)
        # Biaix de vent (opcional, capa vectorial)
        self.wind_bias  = wind_bias if wind_bias is not None else np.zeros_like(humidity)
        # Direcció global del vent (unitària). Format: (x, y)
        if wind_vector is None:
            self.wind_vector = np.array([1.0, 0.0], dtype=float)
        else:
            wv = np.array(wind_vector, dtype=float)
            nrm = np.linalg.norm(wv)
            self.wind_vector = (wv / nrm) if nrm > 0 else np.array([1.0, 0.0], dtype=float)

        self.relief_range = max(1e-6, float(np.max(self.relief) - np.min(self.relief)))
        self.fuel_min = float(np.min(self.fuel))
        self.fuel_range = max(1e-6, float(np.max(self.fuel) - self.fuel_min))

        # ── Comptadors interns ───────────────────────────────────────
        # Temps que porta cada cel·la en estat BURNING (per calcular combustible)
        self._burn_timer = np.zeros((self.rows, self.cols), dtype=float)
        # Temps que porta esperant (relacionat amb humitat) cada cel·la que ha rebut foc
        self._ignition_wait = np.full((self.rows, self.cols), -1.0)

        # ── Historial per a la visualització ─────────────────────────
        self.history = []
        self.time    = 0

    # ── Ignició inicial ─────────────────────────────────────────────
    def ignite(self, row: int, col: int):
        """Inicia el foc en la cel·la (row, col)."""
        if self.vegetation[row, col] > 0:
            self.fire_state[row, col]   = BURNING
            self._burn_timer[row, col]  = 0
            self._ignition_wait[row, col] = 0
            print(f"[🔥] Ignició a ({row},{col}) "
                  f"— vegetació={self.vegetation[row,col]:.1f}h, "
                  f"humitat={self.humidity[row,col]:.1f}h")
        else:
            print(f"[!] La cel·la ({row},{col}) no té vegetació (no crema).")

    # ── Funció de veïnatge (vn) ──────────────────────────────────────
    def _get_neighbors(self, r: int, c: int) -> list[tuple[int,int]]:
        """
        Retorna la llista de cel·les del veïnatge de Moore de (r,c).
        Condició frontera: FIXA (els extrems no es consideren).
        """
        neighbors = []
        for dr, dc in self.MOORE_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def _directional_wind_factor(self,
                                 r: int,
                                 c: int,
                                 burning_neighbors: list[tuple[int, int]]) -> float:
        """
        Calcula el multiplicador de propagació degut al vent.
        Combina la intensitat local (wind_bias) amb l'alineació entre 
        l'avanç del foc i la direcció del vent.
        """
        if not burning_neighbors:
            return 1.0

        wx, wy = self.wind_vector[0], self.wind_vector[1]
        alignments = []

        for nr, nc in burning_neighbors:
            # Vector de propagació: de la cel·la en flames cap a la candidata.
            vx = c - nc
            vy = r - nr
            vnorm = np.hypot(vx, vy)
            if vnorm == 0:
                continue
            
            # Producte escalar (equival a cos(theta) perquè són unitaris)
            # 1 = alineat, 0 = perpendicular, -1 = en contra
            ux, uy = vx / vnorm, vy / vnorm
            align = ux * wx + uy * wy
            alignments.append(float(align))

        if not alignments:
            return 1.0

        # Reutilitzem la regla de majoria (Part 1) per detectar "dominància"
        # de propagació a favor del vent dins el veïnatge local.
        downwind_votes = [1 if a > 0 else 0 for a in alignments]
        downwind_majority = wolfram_majority(downwind_votes)

        # Normalitzem la màxima alineació de [-1, 1] a [0, 1]
        max_align = max(alignments)
        directional_term = 0.5 * (max_align + 1.0)

        # Si hi ha majoria de propagació a favor del vent, garantim un efecte mínim
        if downwind_majority == 1:
            directional_term = max(directional_term, 0.7)

        directional_term = min(1.0, max(0.0, directional_term))
        
        # Fórmula final: factor 1.0 (base) + benefici direccional escalat per la intensitat
        return 1.0 + self.wind_bias[r, c] * directional_term

    def _relief_factor(self,
                       r: int,
                       c: int,
                       burning_neighbors: list[tuple[int, int]]) -> float:
        """Modulador per relleu: pujar pendent accelera, baixar pendent frena."""
        if not burning_neighbors:
            return 1.0

        slopes = []
        z_target = self.relief[r, c]
        for nr, nc in burning_neighbors:
            z_source = self.relief[nr, nc]
            slope = (z_target - z_source) / self.relief_range
            slopes.append(float(np.clip(slope, -1.0, 1.0)))

        max_uphill = max(0.0, max(slopes))
        max_downhill = max(0.0, -min(slopes))
        factor = 1.0 + 0.5 * max_uphill - 0.25 * max_downhill
        return float(np.clip(factor, 0.6, 1.6))

    def _fuel_factor(self, r: int, c: int) -> float:
        """Modulador per combustible local (força de propagació)."""
        f_norm = (self.fuel[r, c] - self.fuel_min) / self.fuel_range
        return 0.75 + 0.75 * float(np.clip(f_norm, 0.0, 1.0))

    # ── Funció d'evolució Λ (un pas de temps) ────────────────────────
    def step(self, dt: float = 1.0):
        """
        Avança el model un pas de temps dt (en hores).

        Lògica de l'evolució:
        ─────────────────────────────────────────────
        1. UNBURNED  → comprova si algun veí és BURNING
             Si sí: inicia el comptador d'ignició (caldrà superar la humitat)
             Si l'espera ≥ humitat: passa a BURNING
        2. BURNING   → decrementa combustible (vegetació)
             Si vegetació ≤ 0: passa a BURNED
        3. BURNED    → estat terminal (no canvia)

        Biaix de vent: augmenta la probabilitat de propagació cap a
        les zones marcades pels polígons vectorials.
        """
        new_state  = self.fire_state.copy()
        new_hum    = self.humidity.copy()
        new_veg    = self.vegetation.copy()
        new_wait   = self._ignition_wait.copy()
        new_timer  = self._burn_timer.copy()

        for r in range(self.rows):
            for c in range(self.cols):

                state = self.fire_state[r, c]

                # ── UNBURNED: pot rebre el foc ──────────────────────
                if state == UNBURNED:
                    neighbors = self._get_neighbors(r, c)
                    burning_neighbors = [
                        (nr, nc) for nr, nc in neighbors
                        if self.fire_state[nr, nc] == BURNING
                    ]
                    has_burning_neighbor = len(burning_neighbors) > 0

                    if has_burning_neighbor:
                        # Vent direccional: combina biaix espacial + direcció.
                        wind_factor = self._directional_wind_factor(r, c, burning_neighbors)
                        relief_factor = self._relief_factor(r, c, burning_neighbors)
                        fuel_factor = self._fuel_factor(r, c)
                        spread_factor = wind_factor * relief_factor * fuel_factor

                        if new_wait[r, c] < 0:
                            # Primera vegada que rep foc: inicia l'espera
                            new_wait[r, c] = 0.0
                        else:
                            new_wait[r, c] += dt * spread_factor

                        # Supera la humitat? → comença a cremar
                        if new_wait[r, c] >= self.humidity[r, c]:
                            if self.vegetation[r, c] > 0:
                                new_state[r, c]  = BURNING
                                new_timer[r, c]  = 0
                                new_hum[r, c]    = 0  # humitat consumida
                            # Si no hi ha vegetació, no crema

                # ── BURNING: consumeix vegetació ────────────────────
                elif state == BURNING:
                    new_veg[r, c]   = max(0.0, self.vegetation[r, c] - dt)
                    new_timer[r, c] = self._burn_timer[r, c] + dt

                    if new_veg[r, c] <= 0:
                        new_state[r, c] = BURNED

                # ── BURNED: estat absorbent ──────────────────────────
                # (no cal fer res)

        # Actualitzem totes les capes
        self.fire_state        = new_state
        self.humidity          = new_hum
        self.vegetation        = new_veg
        self._ignition_wait    = new_wait
        self._burn_timer       = new_timer
        self.time             += dt

        # Guardem snapshot a l'historial
        self.history.append(self.fire_state.copy())

    # ── Execució completa ────────────────────────────────────────────
    def run(self, max_steps: int = 60, dt: float = 1.0,
            verbose: bool = True):
        """
        Executa el model fins que no hi hagi cel·les en flames
        o s'arribi al màxim de passos.
        """
        self.history = [self.fire_state.copy()]

        for step_n in range(1, max_steps + 1):
            self.step(dt)

            n_burning = np.sum(self.fire_state == BURNING)
            n_burned  = np.sum(self.fire_state == BURNED)
            n_unburned= np.sum(self.fire_state == UNBURNED)

            if verbose and step_n % 5 == 0:
                print(f"  t={self.time:5.1f}h  |  "
                      f"Cremant: {n_burning:4d}  "
                      f"Cremat: {n_burned:4d}  "
                      f"Pendent: {n_unburned:4d}")

            if n_burning == 0:
                print(f"\n[✓] Simulació finalitzada al pas {step_n} "
                      f"(t={self.time:.1f}h) — no queden cel·les en flames.")
                break
        else:
            print(f"\n[!] S'ha arribat al màxim de {max_steps} passos.")


# ═══════════════════════════════════════════════════════════════════════
#  VISUALITZACIÓ
# ═══════════════════════════════════════════════════════════════════════

def plot_layers(humidity: np.ndarray,
                vegetation: np.ndarray,
                relief: np.ndarray,
                fuel: np.ndarray,
                wind_bias: np.ndarray | None,
                title: str = "Capes inicials"):
    """Mostra les capes d'entrada del model."""
    ncols = 5 if wind_bias is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(humidity, cmap='Blues', interpolation='nearest')
    axes[0].set_title('E1: Humitat (hores protecció)', fontsize=11)
    plt.colorbar(im0, ax=axes[0], label='hores')

    im1 = axes[1].imshow(vegetation, cmap='Greens', interpolation='nearest')
    axes[1].set_title('E2: Vegetació (hores combustible)', fontsize=11)
    plt.colorbar(im1, ax=axes[1], label='hores')

    # E3 és l'estat del foc (no mostrem la capa ja que només es veuria el punt de ignició inicial)

    im2 = axes[2].imshow(relief, cmap='terrain', interpolation='nearest')
    axes[2].set_title('E4: Relleu / elevació', fontsize=11)
    plt.colorbar(im2, ax=axes[2], label='m')

    im3 = axes[3].imshow(fuel, cmap='YlOrBr', interpolation='nearest')
    axes[3].set_title('E5: Combustible local', fontsize=11)
    plt.colorbar(im3, ax=axes[3], label='força')

    if wind_bias is not None:
        im4 = axes[4].imshow(wind_bias, cmap='Oranges', interpolation='nearest')
        axes[4].set_title('Biaix vent (capa vectorial)', fontsize=11)
        plt.colorbar(im4, ax=axes[4])

    for ax in axes:
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')

    plt.tight_layout()
    return fig


def plot_evolution(ca: WildfireCA, snapshots: list[int] | None = None):
    """
    Mostra l'evolució temporal del camp de foc per un conjunt de passos.
    """
    cmap = mcolors.ListedColormap(['#d4e9b5', '#e84c0e', '#3d3d3d'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)

    n_hist = len(ca.history)
    if snapshots is None:
        # Escollim fins a 8 snapshots equidistants
        idxs = np.linspace(0, n_hist - 1, min(8, n_hist), dtype=int)
        snapshots = list(idxs)

    n = len(snapshots)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4.5, rows * 4),
                              squeeze=False)
    fig.suptitle('Evolució temporal de l\'incendi (m:n-CA^k)',
                 fontsize=14, fontweight='bold')

    for idx, snap_i in enumerate(snapshots):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.imshow(ca.history[snap_i], cmap=cmap, norm=norm,
                  interpolation='nearest')
        ax.set_title(f't = {snap_i:.0f} h', fontsize=10)
        ax.axis('off')

    # Amaguem subplots buits
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    # Llegenda
    legend_patches = [
        mpatches.Patch(color='#d4e9b5', label='UNBURNED (pendent)'),
        mpatches.Patch(color='#e84c0e', label='BURNING (cremant)'),
        mpatches.Patch(color='#3d3d3d', label='BURNED (cremat)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_statistics(ca: WildfireCA):
    """Mostra les estadístiques de la propagació al llarg del temps."""
    times    = list(range(len(ca.history)))
    unburned = [np.sum(s == UNBURNED) for s in ca.history]
    burning  = [np.sum(s == BURNING)  for s in ca.history]
    burned   = [np.sum(s == BURNED)   for s in ca.history]
    total    = ca.rows * ca.cols

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Estadístiques de propagació', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.fill_between(times, unburned, alpha=0.4, color='#4caf50', label='Pendent')
    ax.fill_between(times, burning,  alpha=0.8, color='#e84c0e', label='Cremant')
    ax.fill_between(times, burned,   alpha=0.5, color='#555555', label='Cremat')
    ax.set_xlabel('Temps (hores)')
    ax.set_ylabel('Nombre de cel·les')
    ax.set_title('Nombre de cel·les per estat')
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    pct_burned = [b / total * 100 for b in burned]
    ax2.plot(times, pct_burned, color='#e84c0e', linewidth=2)

    # Reutilització de la Part 1: coarse_grain de la sèrie de cel·les cremades.
    burned_binary_matrix = np.array(
        [(s == BURNED).astype(int).ravel() for s in ca.history], dtype=int
    )
    coarse_burned = wolfram_coarse_grain(burned_binary_matrix, k=4)
    coarse_pct_burned = np.mean(coarse_burned, axis=1) * 100.0
    ax2.plot(times, coarse_pct_burned, color='#8d6e63', linestyle='--', linewidth=1.6,
             label='Tendència coarse (Part 1, K=4)')

    ax2.set_xlabel('Temps (hores)')
    ax2.set_ylabel('% de la superfície')
    ax2.set_title('Superfície cremada acumulada (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_final_state(ca: WildfireCA,
                     humidity_orig: np.ndarray,
                     vegetation_orig: np.ndarray):
    """Mostra l'estat final amb les capes comparades."""
    cmap_fire = mcolors.ListedColormap(['#d4e9b5', '#e84c0e', '#3d3d3d'])
    norm_fire = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap_fire.N)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Resultat final de la simulació', fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(humidity_orig, cmap='Blues', interpolation='nearest')
    axes[0].set_title('E1: Humitat inicial')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(vegetation_orig, cmap='Greens', interpolation='nearest')
    axes[1].set_title('E2: Vegetació inicial')
    plt.colorbar(im1, ax=axes[1])

    axes[2].imshow(ca.fire_state, cmap=cmap_fire, norm=norm_fire,
                   interpolation='nearest')
    axes[2].set_title(f'E3: Estat final del foc (t={ca.time:.0f}h)')
    legend_patches = [
        mpatches.Patch(color='#d4e9b5', label='Pendent'),
        mpatches.Patch(color='#e84c0e', label='Cremant'),
        mpatches.Patch(color='#3d3d3d', label='Cremat'),
    ]
    axes[2].legend(handles=legend_patches, loc='lower right', fontsize=8)

    for ax in axes:
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')

    plt.tight_layout()
    return fig


def plot_wind_diagnostics(ca: WildfireCA,
                          wind_bias: np.ndarray,
                          title: str = 'Diagnosi de vent (biaix + direcció)'):
    """Mostra com el vent afecta la propagació: camp espacial i direcció preferent."""
    fig = plt.figure(figsize=(16, 5.5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── 1) Camp de biaix de vent + quiver amb la direcció global ─────
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(wind_bias, cmap='Oranges', interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='intensitat vent [0..1]')
    ax1.set_title('Camp espacial de vent')
    ax1.set_xlabel('Columna')
    ax1.set_ylabel('Fila')

    rows, cols = wind_bias.shape
    gx = np.linspace(0, cols - 1, 8)
    gy = np.linspace(0, rows - 1, 8)
    xx, yy = np.meshgrid(gx, gy)
    u = np.full_like(xx, ca.wind_vector[0], dtype=float)
    v = np.full_like(yy, -ca.wind_vector[1], dtype=float)
    # 'quiver' dibuixa fletxes per representar un camp de vectors (aquí, la direcció del vent)
    ax1.quiver(xx, yy, u, v, color='black', alpha=0.75, scale=12)

    # ── 2) Pes direccional per a les 8 direccions de Moore ───────────
    ax2 = fig.add_subplot(1, 3, 2)
    labels = ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE']
    weights = []
    wx, wy = ca.wind_vector[0], ca.wind_vector[1]
    for dr, dc in WildfireCA.MOORE_OFFSETS:
        vx, vy = dc, dr
        n = np.hypot(vx, vy)
        ux, uy = vx / n, vy / n
        # Pes positiu quan la propagació està alineada amb el vent.
        score = max(0.0, ux * wx + uy * wy)
        weights.append(score)
    ax2.bar(labels, weights, color='#ef6c00', alpha=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('pes direccional')
    ax2.set_title('Preferència de propagació per direcció')
    ax2.grid(axis='y', alpha=0.25)

    # ── 3) Estat final amb fletxa de direcció del vent ────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    cmap_fire = mcolors.ListedColormap(['#d4e9b5', '#e84c0e', '#3d3d3d'])
    norm_fire = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap_fire.N)
    ax3.imshow(ca.fire_state, cmap=cmap_fire, norm=norm_fire, interpolation='nearest')
    ax3.set_title('Estat final + direcció del vent')
    ax3.set_xlabel('Columna')
    ax3.set_ylabel('Fila')
    cx, cy = cols * 0.15, rows * 0.15
    ax3.arrow(cx, cy,
              ca.wind_vector[0] * cols * 0.2,
              ca.wind_vector[1] * rows * 0.2,
              head_width=0.9, head_length=0.9,
              fc='white', ec='white', linewidth=2)
    ax3.text(cx, cy - 1.5, 'Direcció vent', color='white', fontsize=9,
             bbox=dict(facecolor='black', alpha=0.45, pad=2))

    plt.tight_layout()
    return fig


def run_wind_scenarios(humidity: np.ndarray,
                       vegetation: np.ndarray,
                       relief: np.ndarray,
                       fuel: np.ndarray,
                       ignition: tuple[int, int],
                       max_steps: int = 80,
                       dt: float = 1.0) -> dict:
    """
    Executa diversos escenaris de vent per comparar l'impacte al resultat final.
    Inclou explícitament un cas sense vent.
    """
    rows, cols = humidity.shape

    scenarios = {
        'sense_vent': {
            'bias': np.zeros((rows, cols), dtype=float),
            'vector': np.array([1.0, 0.0], dtype=float),
        },
        'vent_est': {
            'bias': np.full((rows, cols), 0.8, dtype=float),
            'vector': np.array([1.0, 0.0], dtype=float),
        },
        'vent_oest': {
            'bias': np.full((rows, cols), 0.8, dtype=float),
            'vector': np.array([-1.0, 0.0], dtype=float),
        },
        'vent_nord': {
            'bias': np.full((rows, cols), 0.8, dtype=float),
            'vector': np.array([0.0, -1.0], dtype=float),
        },
        'vent_sud_est': {
            'bias': np.full((rows, cols), 0.8, dtype=float),
            'vector': np.array([1.0, 1.0], dtype=float),
        },
    }

    results = {}
    ig_r, ig_c = ignition
    total = rows * cols

    for name, cfg in scenarios.items():
        ca_s = WildfireCA(
            humidity,
            vegetation,
            relief=relief,
            fuel=fuel,
            wind_bias=cfg['bias'],
            wind_vector=cfg['vector'],
        )
        ca_s.ignite(ig_r, ig_c)
        ca_s.run(max_steps=max_steps, dt=dt, verbose=False)

        n_burned = int(np.sum(ca_s.fire_state == BURNED))
        results[name] = {
            'ca': ca_s,
            'burned_pct': n_burned / total * 100.0,
        }

    return results


def plot_wind_scenarios_comparison(results: dict,
                                   title: str = 'Comparació d\'escenaris de vent'):
    """Mostra els estats finals per escenari i el percentatge cremat."""
    names = list(results.keys())
    n = len(names)

    cmap_fire = mcolors.ListedColormap(['#d4e9b5', '#e84c0e', '#3d3d3d'])
    norm_fire = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap_fire.N)

    fig = plt.figure(figsize=(4.4 * n, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, name in enumerate(names):
        ax = fig.add_subplot(2, n, i + 1)
        ca_s = results[name]['ca']
        ax.imshow(ca_s.fire_state, cmap=cmap_fire, norm=norm_fire, interpolation='nearest')
        ax.set_title(f"{name}\ncremat={results[name]['burned_pct']:.1f}%")
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')

    ax_bar = fig.add_subplot(2, 1, 2)
    burned_vals = [results[name]['burned_pct'] for name in names]
    ax_bar.bar(names, burned_vals, color='#e84c0e', alpha=0.85)
    ax_bar.set_ylabel('% cremat')
    ax_bar.set_title('Superfície cremada final per escenari')
    ax_bar.set_ylim(0, 100)
    ax_bar.grid(axis='y', alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  EXECUCIÓ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("  MODEL 5:2-CA^1 — PROPAGACIÓ D'INCENDI FORESTAL")
    print("=" * 65)

    # ── 1. Càrrega de les capes raster IDRISI32 ──────────────────────
    print("\n[1] Carregant capes raster (format IDRISI32)...")
    humidity,   meta_h = load_raster_layer(
        os.path.join(DATA_DIR, 'Initialize.doc'),
        os.path.join(DATA_DIR, 'Initialize.img')
    )
    vegetation, meta_v = load_raster_layer(
        os.path.join(DATA_DIR, 'vegetation.doc'),
        os.path.join(DATA_DIR, 'vegetation.img')
    )
    relief, meta_r = load_raster_layer(
        os.path.join(DATA_DIR, 'relief.doc'),
        os.path.join(DATA_DIR, 'relief.img')
    )
    fuel, meta_f = load_raster_layer(
        os.path.join(DATA_DIR, 'fuel.doc'),
        os.path.join(DATA_DIR, 'fuel.img')
    )

    # ── 2. Càrrega de la capa vectorial (vent) — opcional ─────────────
    print("\n[2] Carregant capa vectorial de vent (IDRISI32 vec/dvc)...")
    try:
        polygons = read_idrisi32_vec(
            os.path.join(DATA_DIR, 'wind.dvc'),
            os.path.join(DATA_DIR, 'wind.vec')
        )
        rows, cols = humidity.shape
        x_min = float(meta_v.get('min. x', 0))
        x_max = float(meta_v.get('max. x', cols))
        y_min = float(meta_v.get('min. y', 0))
        y_max = float(meta_v.get('max. y', rows))
        wind_bias = polygon_to_wind_bias(polygons, rows, cols,
                                         x_min, x_max, y_min, y_max)
        wind_vector = estimate_wind_vector_from_polygons(polygons)
        print(f"    Biaix màxim vent: {wind_bias.max():.2f}")
        print(f"    Direcció vent estimada (x,y): ({wind_vector[0]:+.2f}, {wind_vector[1]:+.2f})")
        if len(polygons) == 0:
            print("    [!] No s'han detectat polígons vectorials; el camp de vent queda uniforme.")
        use_wind = True
    except Exception as e:
        print(f"[!] Capa vectorial no disponible: {e}")
        wind_bias = None
        wind_vector = None
        use_wind  = False

    # ── 3. Visualitzem les capes inicials ─────────────────────────────
    print("\n[3] Visualitzant capes inicials...")
    humidity_orig   = humidity.copy()
    vegetation_orig = vegetation.copy()
    relief_orig = relief.copy()
    fuel_orig = fuel.copy()

    fig_layers = plot_layers(humidity, vegetation, relief, fuel,
                              wind_bias if use_wind else None,
                              "Capes inicials del model")
    fig_layers.savefig(os.path.join(OUTPUT_DIR, 'wildfire_01_layers.png'),
                       dpi=150, bbox_inches='tight')
    print("    Guardat: wildfire_01_layers.png")

    # ── 4. Creació del CA i ignició ───────────────────────────────────
    print("\n[4] Creant l'autòmat cel·lular 5:2-CA^1...")
    ca = WildfireCA(
        humidity,
        vegetation,
        relief=relief,
        fuel=fuel,
        wind_bias=wind_bias if use_wind else None,
        wind_vector=wind_vector if use_wind else None,
    )

    # Punt d'ignició: centre del raster (simulem un llamp o foc initial)
    ig_r, ig_c = humidity.shape[0] // 4, humidity.shape[1] // 4
    print(f"    Punt d'ignició: ({ig_r}, {ig_c})")
    ca.ignite(ig_r, ig_c)

    # ── 5. Execució de la simulació ───────────────────────────────────
    print("\n[5] Executant la simulació (dt=1h)...")
    ca.run(max_steps=80, dt=1.0, verbose=True)

    # ── 6. Visualització de l'evolució ───────────────────────────────
    print("\n[6] Generant gràfics d'evolució...")
    fig_evo = plot_evolution(ca)
    fig_evo.savefig(os.path.join(OUTPUT_DIR, 'wildfire_02_evolution.png'),
                    dpi=150, bbox_inches='tight')
    print("    Guardat: wildfire_02_evolution.png")

    # ── 7. Estadístiques ──────────────────────────────────────────────
    fig_stats = plot_statistics(ca)
    fig_stats.savefig(os.path.join(OUTPUT_DIR, 'wildfire_03_statistics.png'),
                      dpi=150, bbox_inches='tight')
    print("    Guardat: wildfire_03_statistics.png")

    # ── 8. Estat final ────────────────────────────────────────────────
    fig_final = plot_final_state(ca, humidity_orig, vegetation_orig)
    fig_final.savefig(os.path.join(OUTPUT_DIR, 'wildfire_04_final.png'),
                      dpi=150, bbox_inches='tight')
    print("    Guardat: wildfire_04_final.png")

    # ── 9. Diagnosi específica del vent ─────────────────────────────
    if use_wind and wind_bias is not None:
        fig_wind = plot_wind_diagnostics(ca, wind_bias)
        fig_wind.savefig(os.path.join(OUTPUT_DIR, 'wildfire_05_wind_diagnostics.png'),
                         dpi=150, bbox_inches='tight')
        print("    Guardat: wildfire_05_wind_diagnostics.png")

    # ── 10. Comparació de múltiples tipus de vent (inclou sense vent) ──
    print("\n[8C] Comparant escenaris: sense vent + vents direccionals...")
    scenario_results = run_wind_scenarios(
        humidity_orig,
        vegetation_orig,
        relief_orig,
        fuel_orig,
        ignition=(ig_r, ig_c),
        max_steps=80,
        dt=1.0,
    )
    fig_scen = plot_wind_scenarios_comparison(scenario_results)
    fig_scen.savefig(os.path.join(OUTPUT_DIR, 'wildfire_06_wind_scenarios.png'),
                     dpi=150, bbox_inches='tight')
    print("    Guardat: wildfire_06_wind_scenarios.png")

    # ── 11. Resum ──────────────────────────────────────────────────────
    total_cells = ca.rows * ca.cols
    n_burned    = int(np.sum(ca.fire_state == BURNED))
    n_burning   = int(np.sum(ca.fire_state == BURNING))
    n_unburned  = int(np.sum(ca.fire_state == UNBURNED))

    print("\n" + "=" * 65)
    print("  RESUM FINAL DE LA SIMULACIÓ")
    print("=" * 65)
    print(f"  Temps total simulat : {ca.time:.1f} hores")
    print(f"  Mida del raster     : {ca.rows} x {ca.cols} cel·les")
    print(f"  Cel·les totals      : {total_cells}")
    print(f"  Cel·les cremades    : {n_burned:4d}  "
          f"({n_burned/total_cells*100:.1f}%)")
    print(f"  Cel·les cremant     : {n_burning:4d}")
    print(f"  Cel·les sense cremar: {n_unburned:4d}  "
          f"({n_unburned/total_cells*100:.1f}%)")
    print(f"  Vent (capa vectorial): {'Sí' if use_wind else 'No'}")
    print("=" * 65)
    print("\n[✓] Simulació completada. Fitxers guardats a ./output/")

    plt.close('all')
