"""
Real-world validation data: EPA Toxics Release Inventory off-site transfers.

Every model result so far comes from a market this project generated itself.
TRI is the closest public record of what the model actually predicts: a
facility with a waste stream, a facility that took it, the chemical, the
quantity, and what was done with it. File type 3A lists every off-site
transfer by sending facility, receiving facility, chemical and pounds,
reported annually since 1987.

This module turns those files into the exact inputs the MC-GNN was trained on
-- 17 numbers per plant, 8 per candidate pair, in the same order and on the
same scales -- so the production model can be run on them unchanged, and a
second copy of the same architecture can be trained on them for comparison.

Nothing here is invented to fill a gap. Where TRI has no counterpart for an
input the platform uses, the value is the one the platform itself assigns
when it knows nothing: a new company's neutral trust and rating, and the
quality score it gives a listing with no purity on record. Everything else
comes from the reported transfers, and every relationship feature is computed
only from years strictly before the one being scored.

Scope, stated once so it is not repeated as a caveat everywhere else:
  - only recycling and energy-recovery transfers are used; landfill,
    treatment and sewage shipments are not industrial symbiosis
  - TRI records transfers that happened, never ones that were refused, so
    "did not happen" examples are drawn from facilities that could plausibly
    have taken the waste and did not
  - quantities are TRI chemical quantities, not whole waste streams

Data is read from `us data/` in the project root (override with the
TRI_DATA_DIR environment variable): each year either as extracted
`US_3a_<year>.txt` or as the national `us_<year>.zip`, plus the EPA Facility
Registry (`frs_national_single.zip`) and Census ZIP centroids
(`gaz_zcta_2024.zip`) to locate facilities that do not report to TRI
themselves. None of it is committed; see `DOWNLOADS` for where it comes from.
"""

from __future__ import annotations

import csv
import io
import math
import os
import random
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from ai.features.node_features import (
    EXCHANGES_SATURATION,
    INDUSTRY_SECTORS,
    LISTINGS_SATURATION,
    NEUTRAL_RATING,
    NEUTRAL_TRUST,
    REQUIREMENTS_SATURATION,
)
from app.enums.common import QuantityUnit
from app.services.recommendation_service import CATEGORY_MATCH_COMPAT
from app.utils.distance import EARTH_RADIUS_KM, estimate_transport_cost, haversine_distance
from app.utils.edge_encoding import EDGE_FEATURE_NAMES, encode_edge_features
from app.utils.quantity import carbon_saving_for_quantity, quantity_fit_pct, transport_tons

csv.field_size_limit(10_000_000)

DATA_DIR = Path(os.getenv("TRI_DATA_DIR", str(PROJECT_ROOT / "us data")))

DOWNLOADS = {
    "tri_year": "https://www.epa.gov/system/files/other-files/2025-11/us_{year}.zip",
    "frs": "https://ordsext.epa.gov/FLA/www3/state_files/national_single.zip",
    "zip_centroids": (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2024_Gazetteer/2024_Gaz_zcta_national.zip"
    ),
}

# The model's input width, fixed by the checkpoint it was trained into.
NODE_FEATURE_DIM = 7 + len(INDUSTRY_SECTORS)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

# What the platform assigns a listing with no purity on record -- see the
# quality_compat defaults in app/services/recommendation_service.py. TRI never
# records purity, so every pair gets the platform's own "nothing known" value.
NO_PURITY_DATA_QUALITY = 100.0

LB_TO_KG = 0.45359237
GRAM_TO_KG = 0.001

# TRI's recovery management codes (Form R section 6.2): the transfers that put
# waste back to use. Everything else in 3A is disposal or treatment.
RECOVERY_METHODS = ("M20", "M24", "M26", "M28", "M93", "M56", "M92")

# NAICS -> the platform's industry sectors, from the NAICS 2022 definitions.
# Longest prefix wins; anything unlisted falls to "General Manufacturing",
# the same fallback the platform applies to a company whose industry is not
# one of its sectors (plastics and rubber, waste management, utilities...).
NAICS_SECTOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("3254", "Pharmaceuticals"),
    ("3361", "Automotive"),
    ("3362", "Automotive"),
    ("3363", "Automotive"),
    ("311", "Food Processing"),
    ("312", "Food Processing"),
    ("313", "Textiles & Apparel"),
    ("314", "Textiles & Apparel"),
    ("315", "Textiles & Apparel"),
    ("316", "Textiles & Apparel"),
    ("321", "Construction & Materials"),
    ("327", "Construction & Materials"),
    ("23", "Construction & Materials"),
    ("322", "Paper & Packaging"),
    ("323", "Paper & Packaging"),
    ("324", "Chemicals & Petrochemicals"),
    ("325", "Chemicals & Petrochemicals"),
    ("331", "Metal & Metallurgy"),
    ("332", "Metal & Metallurgy"),
    ("334", "Electronics"),
    ("335", "Electronics"),
)
FALLBACK_SECTOR = INDUSTRY_SECTORS[-1]

# How many facilities that did not take a waste stream are sampled for every
# one that did. Gives each query roughly ten candidates -- the size of the
# shortlist the platform shows -- so ranking metrics at 1 and 5 mean something.
NEGATIVES_PER_POSITIVE = 9

# "Could plausibly have taken it" is bounded by how far real transfers
# actually travel: candidates are drawn from within this quantile of the
# distances seen in the history years, measured, not chosen.
CANDIDATE_DISTANCE_QUANTILE = 0.9

SAMPLING_SEED = 2024


# ── small helpers ────────────────────────────────────────────────────────


def normalise_header(name: str) -> str:
    """'  12. OFFSITE  ENERGY RECOVERY M56 LBS' -> 'OFFSITE ENERGY RECOVERY M56 LBS'."""
    name = re.sub(r"^\s*\d+\.\s*", "", name)
    return re.sub(r"\s+", " ", name).strip().upper()


def to_float(value: str) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0.0)
    except ValueError:
        return 0.0


def to_kg(amount: float, unit_of_measure: str) -> float:
    """TRI reports pounds, except dioxins, which are reported in grams."""
    if unit_of_measure.strip().upper().startswith("GRAM"):
        return amount * GRAM_TO_KG
    return amount * LB_TO_KG


def naics_to_sector(naics: str) -> str:
    code = re.sub(r"\D", "", naics or "")
    for prefix, sector in sorted(NAICS_SECTOR_PREFIXES, key=lambda p: -len(p[0])):
        if code.startswith(prefix):
            return sector
    return FALLBACK_SECTOR


def zip5(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[:5] if len(digits) >= 5 else ""


def _name_key(name: str) -> str:
    name = re.sub(r"[^A-Z0-9 ]", "", (name or "").upper())
    name = re.sub(r"\b(LLC|INC|CORP|CORPORATION|CO|LP|LTD|COMPANY|THE)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ── records ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Transfer:
    """One year's recovery shipments of one chemical from one site to another."""

    year: int
    sender: str
    receiver: str
    cas: str
    method: str  # the recovery code carrying most of the mass
    kg: float


@dataclass
class Site:
    key: str
    name: str = ""
    naics: str = ""
    lat: float | None = None
    lon: float | None = None
    loc_source: str = ""  # "tri" | "frs" | "zip" | ""

    @property
    def located(self) -> bool:
        return self.lat is not None and self.lon is not None


@dataclass
class _Receiver:
    key: str
    name: str
    frs: str
    rcra: str
    zip_code: str
    country: str


@dataclass
class YearRead:
    transfers: list[Transfer] = field(default_factory=list)
    senders: dict[str, Site] = field(default_factory=dict)
    receivers: dict[str, _Receiver] = field(default_factory=dict)
    rows: int = 0
    recovery_rows: int = 0


# ── reading TRI ──────────────────────────────────────────────────────────


def _open_3a(year: int, data_dir: Path):
    txt = data_dir / f"US_3a_{year}.txt"
    if txt.exists():
        return open(txt, encoding="utf-8", errors="replace", newline="")
    archive = data_dir / f"us_{year}.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"No TRI file for {year}: expected {txt.name} or {archive.name} in "
            f"{data_dir}. Download it from {DOWNLOADS['tri_year'].format(year=year)}"
        )
    z = zipfile.ZipFile(archive)
    member = next(n for n in z.namelist() if n.upper().startswith("US_3A"))
    return io.TextIOWrapper(z.open(member), encoding="utf-8", errors="replace", newline="")


def read_year(year: int, data_dir: Path = DATA_DIR) -> YearRead:
    """Parse one year of TRI 3A into recovery transfers, keyed by facility."""
    out = YearRead()
    with _open_3a(year, data_dir) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = [normalise_header(h) for h in next(reader)]
        col = {h: i for i, h in enumerate(header)}

        def c(name: str) -> int:
            return col[name]

        method_cols = {}
        for h, i in col.items():
            m = re.search(r"\b(M\d\d)\b", h)
            if m and m.group(1) in RECOVERY_METHODS and h.endswith("LBS"):
                method_cols[m.group(1)] = i

        width = len(header)
        agg: dict[tuple[str, str, str], dict] = {}
        for row in reader:
            if len(row) < width - 1:
                continue
            out.rows += 1
            unit = row[c("UNIT OF MEASURE")]
            by_method = {
                m: to_kg(to_float(row[i]), unit) for m, i in method_cols.items()
            }
            kg = sum(by_method.values())
            if kg <= 0:
                continue
            out.recovery_rows += 1

            frs_s = row[c("FRS FACILITYY ID")].strip()
            sender = f"FRS:{frs_s}" if frs_s else f"TRI:{row[c('TRIFID')].strip()}"
            if sender not in out.senders:
                lat = to_float(row[c("LATITUDE")])
                lon = to_float(row[c("LONGITUDE")])
                out.senders[sender] = Site(
                    key=sender,
                    name=row[c("FACILITY NAME")].strip(),
                    naics=row[c("PRIMARY NAICS CODE")].strip(),
                    lat=lat if lat else None,
                    lon=lon if lon else None,
                    loc_source="tri" if lat and lon else "",
                )

            frs_r = row[c("FRS ID TRANSFER LOCATION")].strip()
            rcra_r = row[c("OFFSITE RCRA ID NR")].strip()
            name_r = row[c("OFFSITE NAME")].strip()
            zip_r = zip5(row[c("OFFSITE ZIPCODE")])
            if frs_r:
                receiver = f"FRS:{frs_r}"
            elif rcra_r:
                receiver = f"RCRA:{rcra_r.upper()}"
            elif name_r:
                receiver = f"NAME:{_name_key(name_r)}|{zip_r}"
            else:
                continue
            if receiver not in out.receivers:
                out.receivers[receiver] = _Receiver(
                    key=receiver,
                    name=name_r,
                    frs=frs_r,
                    rcra=rcra_r,
                    zip_code=zip_r,
                    country=(row[c("OFFSITE COUNTRY ID")].strip() or "US").upper(),
                )

            cas = row[c("CAS NUMBER")].strip() or row[c("CHEMICAL NAME")].strip()
            key = (sender, receiver, cas)
            entry = agg.setdefault(key, {"kg": 0.0, "methods": Counter()})
            entry["kg"] += kg
            for m, v in by_method.items():
                if v > 0:
                    entry["methods"][m] += v

        for (sender, receiver, cas), entry in agg.items():
            if sender == receiver:
                continue
            method = entry["methods"].most_common(1)[0][0]
            out.transfers.append(
                Transfer(year, sender, receiver, cas, method, entry["kg"])
            )
    return out


# ── locating facilities ──────────────────────────────────────────────────


def load_zip_centroids(data_dir: Path = DATA_DIR) -> dict[str, tuple[float, float]]:
    path = data_dir / "gaz_zcta_2024.zip"
    if not path.exists():
        return {}
    z = zipfile.ZipFile(path)
    member = z.namelist()[0]
    out = {}
    with io.TextIOWrapper(z.open(member), encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = [h.strip().upper() for h in next(reader)]
        gi, la, lo = header.index("GEOID"), header.index("INTPTLAT"), header.index("INTPTLONG")
        for row in reader:
            if len(row) > max(gi, la, lo):
                out[row[gi].strip()] = (float(row[la]), float(row[lo]))
    return out


def load_frs_sites(registry_ids: set[str], data_dir: Path = DATA_DIR) -> dict[str, Site]:
    """
    Coordinates and NAICS for the requested FRS registry IDs.

    The national file is ~350 MB zipped, so the rows actually needed are
    cached beside it on first read and reused after that.
    """
    cache = data_dir / "frs_lookup_cache.csv"
    found: dict[str, Site] = {}
    if cache.exists():
        with open(cache, encoding="utf-8", newline="") as fh:
            for rid, lat, lon, naics in csv.reader(fh):
                found[rid] = Site(
                    key=f"FRS:{rid}", naics=naics,
                    lat=float(lat) if lat else None,
                    lon=float(lon) if lon else None,
                    loc_source="frs" if lat and lon else "",
                )
        if registry_ids <= set(found):
            return found

    path = data_dir / "frs_national_single.zip"
    if not path.exists():
        return found
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        # Still downloading, or truncated: the caller falls back to ZIP
        # centroids, and the coverage report says how many that was.
        return found
    member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    with io.TextIOWrapper(z.open(member), encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip().upper() for h in next(reader)]

        def pick(*names: str) -> int | None:
            for n in names:
                if n in header:
                    return header.index(n)
            return None

        ri = pick("REGISTRY_ID")
        la = pick("LATITUDE83", "LATITUDE")
        lo = pick("LONGITUDE83", "LONGITUDE")
        na = pick("NAICS_CODES", "NAICS_CODE")
        wanted = registry_ids - set(found)
        for row in reader:
            if ri is None or len(row) <= ri:
                continue
            rid = row[ri].strip()
            if rid not in wanted:
                continue
            lat = to_float(row[la]) if la is not None else 0.0
            lon = to_float(row[lo]) if lo is not None else 0.0
            naics = row[na].split(",")[0].strip() if na is not None else ""
            found[rid] = Site(
                key=f"FRS:{rid}", naics=naics,
                lat=lat or None, lon=lon or None,
                loc_source="frs" if lat and lon else "",
            )

    with open(cache, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for rid, s in sorted(found.items()):
            w.writerow([rid, s.lat or "", s.lon or "", s.naics])
    return found


def locate_sites(
    reads: list[YearRead], data_dir: Path = DATA_DIR
) -> tuple[dict[str, Site], dict[str, int]]:
    """
    One Site per facility key, located by the most precise source available.

    1. the facility's own TRI report (it filed as a sender in some year)
    2. its entry in the EPA Facility Registry
    3. the centroid of its ZIP code -- a few km off, stated as such
    """
    sites: dict[str, Site] = {}
    for yr in sorted(reads, key=lambda r: min((t.year for t in r.transfers), default=0)):
        for key, s in yr.senders.items():
            if s.located or key not in sites:
                sites[key] = s

    receivers: dict[str, _Receiver] = {}
    for yr in reads:
        receivers.update(yr.receivers)

    frs_needed = {r.frs for k, r in receivers.items() if r.frs and not (k in sites and sites[k].located)}
    frs = load_frs_sites(frs_needed, data_dir)
    centroids = load_zip_centroids(data_dir)

    source = Counter()
    for key, r in receivers.items():
        if key in sites and sites[key].located:
            source["tri"] += 1
            if not sites[key].name:
                sites[key].name = r.name
            continue
        s = Site(key=key, name=r.name)
        hit = frs.get(r.frs) if r.frs else None
        if hit and hit.located:
            s.lat, s.lon, s.naics, s.loc_source = hit.lat, hit.lon, hit.naics, "frs"
        elif r.zip_code in centroids and r.country == "US":
            s.lat, s.lon = centroids[r.zip_code]
            s.loc_source = "zip"
            if hit:
                s.naics = hit.naics
        sites[key] = s
        source[s.loc_source or "unlocated"] += 1
    return sites, dict(source)


# ── the dataset ──────────────────────────────────────────────────────────


class TRIDataset:
    """Recovery transfers plus located sites, ready to be cut into graphs."""

    def __init__(self, transfers: list[Transfer], sites: dict[str, Site]):
        self.transfers = transfers
        self.sites = sites

    @classmethod
    def from_files(cls, years: list[int], data_dir: Path = DATA_DIR) -> tuple["TRIDataset", dict]:
        reads = [read_year(y, data_dir) for y in years]
        sites, receiver_sources = locate_sites(reads, data_dir)
        transfers = [t for r in reads for t in r.transfers]
        per_year = {}
        for y, r in zip(years, reads):
            located = sum(
                1 for t in r.transfers
                if sites.get(t.sender, Site("")).located and sites.get(t.receiver, Site("")).located
            )
            per_year[y] = {
                "rows": r.rows,
                "recovery_rows": r.recovery_rows,
                "transfers": len(r.transfers),
                "both_ends_located": located,
                "located_share": located / len(r.transfers) if r.transfers else 0.0,
            }
        return cls(transfers, sites), {
            "receiver_location_sources": receiver_sources,
            "per_year": per_year,
        }

    def _located(self, key: str) -> bool:
        s = self.sites.get(key)
        return s is not None and s.located

    def _distance(self, a: str, b: str) -> float:
        sa, sb = self.sites[a], self.sites[b]
        return haversine_distance(sa.lat, sa.lon, sb.lat, sb.lon)

    def _distances_from(self, origin: str, keys: list[str]) -> np.ndarray:
        """Great-circle km from one site to many, vectorised (same formula as haversine_distance)."""
        o = self.sites[origin]
        lat = np.radians([self.sites[k].lat for k in keys])
        lon = np.radians([self.sites[k].lon for k in keys])
        lat0, lon0 = math.radians(o.lat), math.radians(o.lon)
        a = (np.sin((lat - lat0) / 2) ** 2
             + math.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2)
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def build_graph(
        self,
        target_year: int,
        *,
        history_start: int | None = None,
        negatives_per_positive: int = NEGATIVES_PER_POSITIVE,
        distance_quantile: float = CANDIDATE_DISTANCE_QUANTILE,
        seed: int = SAMPLING_SEED,
    ) -> dict:
        """
        The graph the model sees when asked about `target_year`.

        History is every transfer before the target year (from `history_start`
        if given). Plant inputs, relationship inputs, the candidate pool and
        the message-passing graph are all built from history alone; the
        target year contributes only the questions -- which facility had
        which waste -- and their answers.
        """
        rng = random.Random(seed)
        hist = [
            t for t in self.transfers
            if t.year < target_year
            and (history_start is None or t.year >= history_start)
            and self._located(t.sender) and self._located(t.receiver)
        ]
        target = [
            t for t in self.transfers
            if t.year == target_year and self._located(t.sender) and self._located(t.receiver)
        ]

        pair_count: Counter = Counter()
        recv_cas: dict[str, set] = defaultdict(set)
        recv_method: dict[str, set] = defaultdict(set)
        by_cas: dict[str, set] = defaultdict(set)
        by_method: dict[str, set] = defaultdict(set)
        intake_cas: dict[tuple, list] = defaultdict(list)
        intake_method: dict[tuple, list] = defaultdict(list)
        intake_all: dict[str, list] = defaultdict(list)
        sent_streams: dict[str, set] = defaultdict(set)
        recv_streams: dict[str, set] = defaultdict(set)
        activity: Counter = Counter()
        popularity: Counter = Counter()
        hist_dist = []
        for t in hist:
            pair_count[(t.sender, t.receiver)] += 1
            recv_cas[t.receiver].add(t.cas)
            recv_method[t.receiver].add(t.method)
            by_cas[t.cas].add(t.receiver)
            by_method[t.method].add(t.receiver)
            intake_cas[(t.receiver, t.cas)].append(t.kg)
            intake_method[(t.receiver, t.method)].append(t.kg)
            intake_all[t.receiver].append(t.kg)
            sent_streams[t.sender].add(t.cas)
            recv_streams[t.receiver].add(t.cas)
            activity[t.sender] += 1
            activity[t.receiver] += 1
            popularity[t.receiver] += 1
            hist_dist.append(self._distance(t.sender, t.receiver))
        distance_cap = (
            float(np.quantile(hist_dist, distance_quantile)) if hist_dist else math.inf
        )

        queries: dict[tuple[str, str], dict] = {}
        for t in target:
            q = queries.setdefault((t.sender, t.cas), {"recv": set(), "kg": 0.0, "methods": Counter()})
            q["recv"].add(t.receiver)
            q["kg"] += t.kg
            q["methods"][t.method] += t.kg

        stats = Counter()
        rows: list[tuple] = []  # (sender, receiver, label, group, feature list, popularity)
        group = 0
        for (sender, cas), q in sorted(queries.items()):
            stats["queries_total"] += 1
            stats["positives_total"] += len(q["recv"])
            method = q["methods"].most_common(1)[0][0]
            pool = (by_cas.get(cas, set()) | by_method.get(method, set())) - {sender}
            positives = sorted(q["recv"] & pool)
            stats["positives_cold"] += len(q["recv"] - pool)
            if not positives:
                continue
            candidates = sorted(pool - q["recv"])
            if not candidates:
                stats["queries_no_alternative"] += 1
                continue
            d = self._distances_from(sender, candidates)
            near = [r for r, km in zip(candidates, d) if km <= distance_cap]
            far = [r for r, km in zip(candidates, d) if km > distance_cap]
            need = negatives_per_positive * len(positives)
            negatives = rng.sample(near, min(need, len(near)))
            if len(negatives) < need and far:
                negatives += rng.sample(far, min(need - len(negatives), len(far)))

            # One shipment size per query: what this facility had of this
            # waste, per partner it placed it with. Positives and negatives
            # are scored against the same number, so size cannot leak which
            # receivers were real.
            kg = q["kg"] / len(q["recv"])
            for receiver, label in [(r, 1.0) for r in positives] + [(r, 0.0) for r in negatives]:
                rows.append((
                    sender, receiver, label, group,
                    self._edge_features(sender, receiver, cas, method, kg,
                                        pair_count, recv_cas, intake_cas,
                                        intake_method, intake_all),
                    popularity[receiver],
                ))
            stats["queries_used"] += 1
            stats["positives_used"] += len(positives)
            stats["negatives_used"] += len(negatives)
            group += 1

        nodes = sorted(
            {t.sender for t in hist} | {t.receiver for t in hist}
            | {r[0] for r in rows} | {r[1] for r in rows}
        )
        index = {k: i for i, k in enumerate(nodes)}
        x = np.array([
            self._node_features(k, sent_streams, recv_streams, activity) for k in nodes
        ], dtype=np.float32).reshape(len(nodes), NODE_FEATURE_DIM)

        msg_pairs = sorted({(t.sender, t.receiver) for t in hist})
        return {
            "target_year": target_year,
            "node_keys": nodes,
            "x": x,
            "edge_index": np.array(
                [[index[r[0]] for r in rows], [index[r[1]] for r in rows]], dtype=np.int64
            ).reshape(2, len(rows)),
            "edge_attr": np.array([r[4] for r in rows], dtype=np.float32).reshape(len(rows), EDGE_FEATURE_DIM),
            "y": np.array([r[2] for r in rows], dtype=np.float32),
            "group_ids": np.array([r[3] for r in rows], dtype=np.int64),
            "receiver_popularity": np.array([r[5] for r in rows], dtype=np.float32),
            "message_edge_index": np.array(
                [[index[a] for a, _ in msg_pairs], [index[b] for _, b in msg_pairs]], dtype=np.int64
            ).reshape(2, len(msg_pairs)),
            "stats": {
                **dict(stats),
                "history_transfers": len(hist),
                "nodes": len(nodes),
                "message_edges": len(msg_pairs),
                "candidate_distance_cap_km": distance_cap,
            },
        }

    def _edge_features(self, sender, receiver, cas, method, kg, pair_count,
                       recv_cas, intake_cas, intake_method, intake_all) -> list[float]:
        distance = self._distance(sender, receiver)
        material = 100.0 if cas in recv_cas.get(receiver, ()) else CATEGORY_MATCH_COMPAT
        typical = (
            intake_cas.get((receiver, cas))
            or intake_method.get((receiver, method))
            or intake_all.get(receiver)
        )
        quantity = quantity_fit_pct(
            kg, QuantityUnit.KG,
            statistics.median(typical) if typical else 0.0, QuantityUnit.KG,
        )
        tons = transport_tons(kg, QuantityUnit.KG)
        prior = pair_count.get((sender, receiver), 0)
        return encode_edge_features(
            distance_km=distance,
            material_compatibility=material,
            quantity_compatibility=quantity,
            quality_compatibility=NO_PURITY_DATA_QUALITY,
            transport_cost=estimate_transport_cost(distance, tons),
            carbon_saving=carbon_saving_for_quantity(kg, QuantityUnit.KG, None),
            prior_successes=prior,
            has_prior_interaction=prior > 0,
        )

    def _node_features(self, key, sent_streams, recv_streams, activity) -> list[float]:
        """
        The platform's 17 plant inputs, in the platform's order and scales.

        TRI has no trust score and no reviews, so both take the value the
        platform gives a company it knows nothing about. The two stock counts
        become the facility's distinct waste streams: chemicals it has sent
        for recovery stand for what it lists, chemicals it has taken in for
        what it requires.
        """
        s = self.sites[key]
        sector = [0.0] * len(INDUSTRY_SECTORS)
        sector[INDUSTRY_SECTORS.index(naics_to_sector(s.naics))] = 1.0
        return [
            s.lat / 90.0,
            s.lon / 180.0,
            NEUTRAL_TRUST,
            min(len(sent_streams.get(key, ())) / LISTINGS_SATURATION, 1.0),
            min(len(recv_streams.get(key, ())) / REQUIREMENTS_SATURATION, 1.0),
            min(activity.get(key, 0) / EXCHANGES_SATURATION, 1.0),
            NEUTRAL_RATING,
        ] + sector

    def market_profile(self, years: list[int]) -> dict:
        """What real recovery trade looks like -- for calibrating the simulator."""
        ts = [t for t in self.transfers if t.year in years
              and self._located(t.sender) and self._located(t.receiver)]
        if not ts:
            return {}
        dist = np.array([self._distance(t.sender, t.receiver) for t in ts])
        kg = np.array([t.kg for t in ts])
        per_pair = Counter((t.sender, t.receiver) for t in ts)
        inbound = Counter(t.receiver for t in ts)
        top20 = sum(v for _, v in inbound.most_common(20))
        pct = lambda a: {f"p{q}": round(float(np.percentile(a, q)), 1) for q in (10, 25, 50, 75, 90)}
        return {
            "years": years,
            "transfers": len(ts),
            "distance_km": pct(dist),
            "kg_per_transfer": pct(kg),
            "pairs_trading_more_than_one_record": round(
                sum(1 for v in per_pair.values() if v > 1) / len(per_pair), 3),
            "share_into_top20_receivers": round(top20 / len(ts), 3),
            "recovery_method_mix": {
                m: round(n / len(ts), 3)
                for m, n in Counter(t.method for t in ts).most_common()
            },
        }
