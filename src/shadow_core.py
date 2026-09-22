import numpy as np
from ripser import ripser
from typing import List, Tuple, Dict, Any
from .core import EpistemicState, TopologicalCoprocessor

class ShadowTopologicalCoprocessor(TopologicalCoprocessor):
    """
    DTC Coprocessor with Zero-Overhead Shadow Telemetry (Kouta Intermediate Fork).
    
    Intervention logic remains 100% identical to DTC v2.0 (state transitions driven purely by H1 density).
    Additional telemetry metrics (Radius of Gyration, Step Velocities, Local Lyapunov Exponent)
    are forked directly from the intermediate distance matrix D, adding < 0.3ms total overhead at N=16.
    """
    def __init__(
        self,
        window_size: int = 16,
        h1_persistence_threshold: float = 0.04,
        density_threshold: float = 0.10,
        lookahead_steps: int = 2,
        hesitation_rg_threshold: float = 0.45
    ):
        super().__init__(window_size, h1_persistence_threshold, density_threshold, lookahead_steps)
        self.hesitation_rg_threshold = hesitation_rg_threshold

    def compute_persistence_with_shadow_fork(self, embeddings: np.ndarray) -> Tuple[int, float, np.ndarray, Dict[str, Any]]:
        N = len(embeddings)
        if N < 4:
            empty_telemetry = {
                "max_life": 0.0,
                "rg": 0.0,
                "mean_velocity": 0.0,
                "terminal_velocity": 0.0,
                "terminal_accel": 0.0,
                "lyapunov_max": 0.0,
                "is_hesitation": False,
                "is_premature_collapse": False
            }
            return 0, 0.0, np.empty((0, 2)), empty_telemetry

        # --- Intermediate Fork Base: Pairwise Distance Matrix D ---
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms
        cos_sim = np.dot(normed, normed.T)
        D = np.sqrt(np.maximum(2.0 - 2.0 * cos_sim, 0.0))

        # --- Core DTC: Ripser H1 from Precomputed Distance Matrix ---
        res = ripser(D, maxdim=1, distance_matrix=True)
        dgm1 = res['dgms'][1]
        if len(dgm1) == 0:
            valid_cycles = []
            density = 0.0
            max_life = 0.0
        else:
            lifetimes = dgm1[:, 1] - dgm1[:, 0]
            valid_cycles = lifetimes[lifetimes > self.h1_threshold]
            density = float(len(valid_cycles)) / float(N)
            max_life = float(np.max(lifetimes))

        # --- Fork ①: Velocity & Premature Collapse (Superdiagonal D) ---
        velocities = np.diag(D, k=1)
        mean_vel = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
        term_vel = float(velocities[-1]) if len(velocities) > 0 else 0.0
        accel = np.diff(velocities) if len(velocities) > 1 else np.array([0.0])
        term_accel = float(accel[-1]) if len(accel) > 0 else 0.0
        is_premature_collapse = bool(term_vel < 0.15 and term_accel < -0.3)

        # --- Fork ②: Radius of Gyration Rg (Hesitation / Foot-Dragging) ---
        # Rg^2 = 1/(2N^2) * sum(D^2)
        rg = float(np.sqrt(np.sum(D**2) / (2.0 * (N**2))))
        is_hesitation = bool(rg < self.hesitation_rg_threshold and mean_vel < 0.50)

        # --- Fork ③: Robust Local Maximal Lyapunov Exponent (Rosenstein method) ---
        # With N=16, we have a sufficiently large window to find true temporal nearest neighbors
        divergences = []
        theiler_window = 2  # Exclude immediate adjacent points [i-1, i, i+1]
        for i in range(N - 2):
            # Candidate indices j that satisfy |i - j| > theiler_window
            candidates = [j for j in range(N - 1) if abs(i - j) > theiler_window]
            if not candidates:
                continue
            # Find nearest neighbor in phase space
            j_min = candidates[int(np.argmin(D[i, candidates]))]
            d0 = D[i, j_min]
            d1 = D[i + 1, j_min + 1]
            if d0 > 1e-4 and d1 > 1e-4:
                divergences.append(np.log(d1 / d0))

        lyapunov_max = float(np.mean(divergences)) if len(divergences) >= 3 else 0.0

        shadow_telemetry = {
            "max_life": round(max_life, 4),
            "rg": round(rg, 4),
            "mean_velocity": round(mean_vel, 4),
            "terminal_velocity": round(term_vel, 4),
            "terminal_accel": round(term_accel, 4),
            "lyapunov_max": round(lyapunov_max, 4),
            "is_hesitation": is_hesitation,
            "is_premature_collapse": is_premature_collapse
        }

        return len(valid_cycles), density, dgm1, shadow_telemetry

    def step(self, new_embedding: np.ndarray) -> Dict[str, Any]:
        self.history_embeddings.append(new_embedding)
        window = np.array(self.history_embeddings[-self.window_size:])
        cycles, density, dgm, shadow = self.compute_persistence_with_shadow_fork(window)
        
        triggered_abort = False
        if density >= self.density_threshold:
            if self.state == EpistemicState.GROUNDED:
                self.state = EpistemicState.WATCHLIST
                self.watchlist_counter = 1
            elif self.state == EpistemicState.WATCHLIST:
                self.watchlist_counter += 1
                if self.watchlist_counter >= self.lookahead_steps:
                    self.state = EpistemicState.DEADLOCK
                    triggered_abort = True
        else:
            if self.state == EpistemicState.WATCHLIST:
                self.state = EpistemicState.GROUNDED
                self.watchlist_counter = 0

        names = {0: 'GROUNDED', 1: 'DEADLOCK', 2: 'WATCHLIST'}
        
        return {
            'step': len(self.history_embeddings),
            'cycles': cycles,
            'density': density,
            'state': self.state,
            'state_name': names[self.state],
            'abort': triggered_abort,
            'shadow_telemetry': shadow
        }
