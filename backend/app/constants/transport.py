"""
Transportation-related configuration.
"""

# Default diesel price (can be updated periodically)
DEFAULT_FUEL_PRICE = 95.0

# Average loading/unloading time
DEFAULT_LOADING_TIME_HOURS = 2

# Logistics constraints
MAX_ROUTE_DISTANCE_KM = 1000

MIN_ROUTE_DISTANCE_KM = 1

# Buffer applied to transport estimates
TRANSPORT_BUFFER_PERCENTAGE = 10