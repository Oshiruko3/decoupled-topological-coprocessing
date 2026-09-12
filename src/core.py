import numpy as np
from ripser import ripser
from typing import List, Tuple, Dict, Any

class EpistemicState:
    GROUNDED = 0
    WATCHLIST = 2
    DEADLOCK = 1

class TopologicalCoprocessor:
    def __init__(self, window_size: int = 16, h1_persistence_threshold: float = 0.04, density_threshold: float = 0.10, lookahead_steps: int = 2):
        self.window_size = window_size
        self.h1_threshold = h1_persistence_threshold
        self.density_threshold = density_threshold
        self.lookahead_steps = lookahead_steps
        self.state = EpistemicState.GROUNDED
        self.watchlist_counter = 0
        self.history_embeddings: List[np.ndarray] = []

    def compute_persistence(self, embeddings: np.ndarray) -> Tuple[int, float, np.ndarray]:
        if len(embeddings) < 4:
            return 0, 0.0, np.empty((0, 2))
        res = ripser(embeddings, maxdim=1)
        dgm1 = res['dgms'][1]
        if len(dgm1) == 0:
            return 0, 0.0, dgm1
        lifetimes = dgm1[:, 1] - dgm1[:, 0]
        valid_cycles = lifetimes[lifetimes > self.h1_threshold]
        density = len(valid_cycles) / len(embeddings)
        return len(valid_cycles), float(density), dgm1

    def step(self, new_embedding: np.ndarray) -> Dict[str, Any]:
        self.history_embeddings.append(new_embedding)
        window = np.array(self.history_embeddings[-self.window_size:])
        cycles, density, dgm = self.compute_persistence(window)
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
        return {'step': len(self.history_embeddings), 'cycles': cycles, 'density': density, 'state': self.state, 'state_name': names[self.state], 'abort': triggered_abort}

    def reset(self):
        self.state = EpistemicState.GROUNDED
        self.watchlist_counter = 0
        self.history_embeddings.clear()
