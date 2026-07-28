from enum import Enum


class VehicleType(str, Enum):
    MINI_TRUCK = "Mini Truck"
    LIGHT_TRUCK = "Light Truck"
    MEDIUM_TRUCK = "Medium Truck"
    HEAVY_TRUCK = "Heavy Truck"
    CONTAINER_TRUCK = "Container Truck"