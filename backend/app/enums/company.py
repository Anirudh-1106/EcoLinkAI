from enum import Enum


class VerificationStatus(str, Enum):
    PENDING = "Pending"
    VERIFIED = "Verified"
    REJECTED = "Rejected"


class DocumentType(str, Enum):
    GST_CERTIFICATE = "GST Certificate"
    POLLUTION_CONTROL_LICENSE = "Pollution Control License"
    FACTORY_LICENSE = "Factory License"
    COMPANY_REGISTRATION = "Company Registration"
    ISO_CERTIFICATE = "ISO Certificate"
    OTHER = "Other"