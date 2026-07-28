from enum import Enum


class WasteStatus(str, Enum):
    AVAILABLE = "Available"
    RESERVED = "Reserved"
    EXPIRED = "Expired"
    EXCHANGED = "Exchanged"


class RequirementStatus(str, Enum):
    OPEN = "Open"
    MATCHED = "Matched"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"


class ExchangeRequestStatus(str, Enum):
    PENDING = "Pending"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"
    CANCELLED = "Cancelled"
    EXPIRED = "Expired"


class ExchangeStatus(str, Enum):
    INITIATED = "Initiated"
    IN_TRANSIT = "In Transit"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class ShipmentStatus(str, Enum):
    PENDING = "Pending"
    PICKED_UP = "Picked Up"
    IN_TRANSIT = "In Transit"
    DELIVERED = "Delivered"
    CANCELLED = "Cancelled"