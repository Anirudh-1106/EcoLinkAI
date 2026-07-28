"""
AI Recommendation Engine Configuration.
"""

# Minimum AI confidence score required
DEFAULT_AI_THRESHOLD = 0.75

# Number of recommendations returned
DEFAULT_MATCH_LIMIT = 10

# Maximum recommendations generated before filtering
MAX_MATCH_CANDIDATES = 100

# Weighting factors for MC-GNN scoring
COMPATIBILITY_WEIGHT = 0.40
DISTANCE_WEIGHT = 0.20
TRANSPORT_COST_WEIGHT = 0.15
CARBON_WEIGHT = 0.15
TRUST_SCORE_WEIGHT = 0.10