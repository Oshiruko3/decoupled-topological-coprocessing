from .core import TopologicalCoprocessor, EpistemicState
from .shadow_core import ShadowTopologicalCoprocessor
from .decision_matrix import CognitivePattern, DecisionAction, diagnose_cognitive_state
from .embeddings import MiniLMEmbeddingProvider
from .projector import ManifoldProjector

__version__ = '3.0.0-dev'
__all__ = [
    'TopologicalCoprocessor', 
    'ShadowTopologicalCoprocessor', 
    'EpistemicState', 
    'CognitivePattern',
    'DecisionAction',
    'diagnose_cognitive_state',
    'MiniLMEmbeddingProvider', 
    'ManifoldProjector'
]

