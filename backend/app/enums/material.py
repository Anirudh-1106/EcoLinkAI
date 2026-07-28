from enum import Enum


class MaterialCategory(str, Enum):
    METAL = "Metal"
    PLASTIC = "Plastic"
    CHEMICAL = "Chemical"
    GLASS = "Glass"
    PAPER = "Paper"
    ORGANIC = "Organic"
    CONSTRUCTION = "Construction"
    ELECTRONIC = "Electronic"
    TEXTILE = "Textile"
    RUBBER = "Rubber"
    WOOD = "Wood"
    OTHER = "Other"



class HazardClass(str, Enum):
    NON_HAZARDOUS = "Non-Hazardous"
    HAZARDOUS = "Hazardous"
    TOXIC = "Toxic"
    FLAMMABLE = "Flammable"
    CORROSIVE = "Corrosive"
    BIOLOGICAL = "Biological"