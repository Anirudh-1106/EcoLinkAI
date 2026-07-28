from enum import Enum


class QuantityUnit(str, Enum):
    KG = "kg"
    TON = "ton"
    LITER = "liter"
    CUBIC_METER = "cubic_meter"
    PIECE = "piece"
    METER = "meter"


class Currency(str, Enum):
    INR = "INR"


class DistanceUnit(str, Enum):
    KILOMETER = "km"