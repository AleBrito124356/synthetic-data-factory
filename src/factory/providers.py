"""Realistic value providers implemented in pure Python.

No third-party faker, no network. Every provider takes a seeded
``random.Random`` so output is fully deterministic for a given seed. The word
lists are intentionally compact but varied enough to look believable in a demo
or a test fixture. Everything produced here is fictional — see the ethics note
in the README.
"""
from __future__ import annotations

import datetime as _dt
import random
import re
import unicodedata
from typing import List, Optional

FIRST_NAMES_F = [
    "Sofia", "Valentina", "Camila", "Isabella", "Lucia", "Mariana", "Gabriela",
    "Daniela", "Ana", "Carmen", "Emma", "Olivia", "Ava", "Mia", "Chloe",
    "Amara", "Yuki", "Mei", "Priya", "Aisha", "Fatima", "Nina", "Elena",
    "Clara", "Ingrid", "Noor", "Leila", "Sara", "Zoe", "Hana",
]

FIRST_NAMES_M = [
    "Mateo", "Santiago", "Diego", "Sebastian", "Nicolas", "Alejandro", "Carlos",
    "Javier", "Miguel", "Andres", "Liam", "Noah", "Ethan", "Lucas", "Mason",
    "Kenji", "Wei", "Arjun", "Omar", "Youssef", "Ivan", "Marco", "Tomas",
    "Felix", "Lars", "Amir", "Karim", "Hassan", "Pablo", "Adrian",
]

LAST_NAMES = [
    "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Perez", "Sanchez", "Ramirez", "Torres", "Flores", "Rivera", "Gomez",
    "Diaz", "Reyes", "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Miller", "Davis", "Wilson", "Anderson", "Nguyen", "Kim", "Patel",
    "Chen", "Wang", "Kumar", "Okafor", "Haddad", "Novak", "Muller", "Rossi",
]

CITIES = [
    "Panama City", "Bogota", "Mexico City", "Lima", "Santiago", "Buenos Aires",
    "Madrid", "Barcelona", "Lisbon", "Paris", "Berlin", "Amsterdam", "Toronto",
    "New York", "Austin", "Seattle", "Miami", "Chicago", "London", "Dublin",
    "Tokyo", "Singapore", "Bengaluru", "Dubai", "Nairobi", "Cape Town",
]

COUNTRIES = [
    "Panama", "Colombia", "Mexico", "Peru", "Chile", "Argentina", "Spain",
    "Portugal", "France", "Germany", "Netherlands", "Canada", "United States",
    "United Kingdom", "Ireland", "Japan", "Singapore", "India", "Kenya",
]

STREETS = [
    "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Lincoln",
    "Sunset", "Riverside", "Bolivar", "Balboa", "Central", "Market", "Union",
    "Highland", "Park", "Lake", "Hill", "Bay",
]

STREET_SUFFIX = ["St", "Ave", "Blvd", "Rd", "Ln", "Dr", "Way", "Ct"]

COMPANIES = [
    "Northwind", "Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay",
    "Soylent", "Stark", "Wayne", "Wonka", "Cyberdyne", "Tyrell", "Aperture",
    "Pied Piper", "Massive Dynamic", "Gekko", "Prestige", "Bluth", "Dunder",
]

COMPANY_SUFFIX = ["Labs", "Systems", "Group", "Technologies", "Solutions", "Analytics", "Digital", "Works"]

JOBS = [
    "Software Engineer", "Data Analyst", "Product Manager", "UX Designer",
    "DevOps Engineer", "Account Executive", "Marketing Manager", "Recruiter",
    "Financial Analyst", "Operations Lead", "Support Specialist", "QA Engineer",
    "Solutions Architect", "Data Scientist", "Nurse", "Physician", "Teacher",
]

# Only RFC 2606 reserved domains (and their subdomains) so no address is ever
# routable or owned by a real party.
EMAIL_DOMAINS = [
    "example.com", "example.org", "example.net", "mail.example.com",
    "demo.example.org", "test.example.net", "inbox.example.com",
]

# RFC 2606 / RFC 6761 names that can never belong to a real party. Used by the
# validator's strict email check.
RESERVED_EMAIL_SUFFIXES = (
    "example.com", "example.org", "example.net",
    ".example", ".test", ".invalid", ".localhost",
)

# RFC 5737 documentation networks (TEST-NET-1/2/3): 3 x 254 usable hosts.
IPV4_DOC_NETWORKS = ("192.0.2", "198.51.100", "203.0.113")

# NANP area codes used for fictional phone numbers. The 555-0100..555-0199
# exchange block is reserved for fictional use in every NANP area code, so
# "+1-AAA-555-01XX" is never a real subscriber line.
NANP_AREA_CODES = [
    a for a in range(201, 990)
    if a % 100 // 10 != 9          # middle digit 9 is reserved for expansion
    and a % 100 != 11              # N11 codes are service codes (411, 911, ...)
    and a not in (555, )           # 555 is not an assignable area code
]

# Lorem-style word pool for the ``text`` provider.
LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo "
    "consequat duis aute irure reprehenderit voluptate velit esse cillum"
).split()


class Providers:
    """Bundle of seeded generators. One instance per generation run."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    # ---- people ----------------------------------------------------------
    def first_name(self) -> str:
        pool = FIRST_NAMES_F if self.rng.random() < 0.5 else FIRST_NAMES_M
        return self.rng.choice(pool)

    def last_name(self) -> str:
        return self.rng.choice(LAST_NAMES)

    def full_name(self) -> str:
        return f"{self.first_name()} {self.last_name()}"

    def email(self, name: Optional[str] = None) -> str:
        if name:
            parts = [p for p in str(name).lower().replace(".", " ").split() if p]
            local = ".".join(parts) if parts else self._slug()
        else:
            local = f"{self.first_name().lower()}.{self.last_name().lower()}"
        # Fold accents (José -> jose), keep only characters that are valid in
        # an unquoted local-part, and never start/end with or repeat a dot.
        local = unicodedata.normalize("NFKD", local)
        local = "".join(c for c in local if (c.isascii() and c.isalnum()) or c in "._-")
        local = ".".join(p for p in local.split(".") if p) or "user"
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    def phone(self) -> str:
        # NANP fictional block: 555-0100..555-0199 is reserved for fiction in
        # every area code, so these can never reach a real subscriber.
        area = self.rng.choice(NANP_AREA_CODES)
        return f"+1-{area}-555-01{self.rng.randint(0, 99):02d}"

    def job(self) -> str:
        return self.rng.choice(JOBS)

    # ---- places ----------------------------------------------------------
    def city(self) -> str:
        return self.rng.choice(CITIES)

    def country(self) -> str:
        return self.rng.choice(COUNTRIES)

    def address(self) -> str:
        num = self.rng.randint(1, 9999)
        return f"{num} {self.rng.choice(STREETS)} {self.rng.choice(STREET_SUFFIX)}"

    # ---- org -------------------------------------------------------------
    def company(self) -> str:
        return f"{self.rng.choice(COMPANIES)} {self.rng.choice(COMPANY_SUFFIX)}"

    # ---- web -------------------------------------------------------------
    def url(self) -> str:
        slug = self._slug()
        return f"https://{slug}.example.com"

    def ipv4(self) -> str:
        # RFC 5737 documentation ranges — reserved, never assigned to a host.
        return f"{self.rng.choice(IPV4_DOC_NETWORKS)}.{self.rng.randint(1, 254)}"

    # ---- text ------------------------------------------------------------
    def sentence(self, min_words: int = 5, max_words: int = 14) -> str:
        n = self.rng.randint(min_words, max_words)
        words = [self.rng.choice(LOREM) for _ in range(n)]
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    def text(self, sentences: int = 2) -> str:
        return " ".join(self.sentence() for _ in range(max(1, sentences)))

    # ---- time ------------------------------------------------------------
    def date_between(self, start: _dt.date, end: _dt.date) -> _dt.date:
        span = (end - start).days
        if span <= 0:
            return start
        return start + _dt.timedelta(days=self.rng.randint(0, span))

    def datetime_between(self, start: _dt.datetime, end: _dt.datetime) -> _dt.datetime:
        span = int((end - start).total_seconds())
        if span <= 0:
            return start
        return start + _dt.timedelta(seconds=self.rng.randint(0, span))

    # ---- helpers ---------------------------------------------------------
    def _slug(self) -> str:
        return (self.rng.choice(COMPANIES) + self.rng.choice(COMPANY_SUFFIX)).lower()


# --------------------------------------------------------------------------
# finite value domains (used for uniqueness planning)
# --------------------------------------------------------------------------
def domain_size(field_type: str) -> Optional[int]:
    """How many distinct values a provider-backed type can produce, or None
    when the space is effectively unbounded. Lets the schema parser reject
    ``unique: true`` on 500 rows of a 19-value ``country`` field up front."""
    sizes = {
        "bool": 2,
        "first_name": len(set(FIRST_NAMES_F + FIRST_NAMES_M)),
        "last_name": len(set(LAST_NAMES)),
        "name": len(set(FIRST_NAMES_F + FIRST_NAMES_M)) * len(set(LAST_NAMES)),
        "city": len(set(CITIES)),
        "country": len(set(COUNTRIES)),
        "job": len(set(JOBS)),
        "company": len(set(COMPANIES)) * len(set(COMPANY_SUFFIX)),
        "ipv4": len(IPV4_DOC_NETWORKS) * 254,
        "phone": len(NANP_AREA_CODES) * 100,
    }
    return sizes.get(field_type)


def enumerate_domain(field_type: str) -> Optional[List[object]]:
    """Every value a small-domain type can take, in a stable order (or None)."""
    first = sorted(set(FIRST_NAMES_F + FIRST_NAMES_M))
    if field_type == "bool":
        return [False, True]
    if field_type == "first_name":
        return list(first)
    if field_type == "last_name":
        return sorted(set(LAST_NAMES))
    if field_type == "name":
        return [f"{f} {l}" for f in first for l in sorted(set(LAST_NAMES))]
    if field_type == "city":
        return sorted(set(CITIES))
    if field_type == "country":
        return sorted(set(COUNTRIES))
    if field_type == "job":
        return sorted(set(JOBS))
    if field_type == "company":
        return [f"{c} {s}" for c in sorted(set(COMPANIES)) for s in sorted(set(COMPANY_SUFFIX))]
    if field_type == "ipv4":
        return [f"{net}.{h}" for net in IPV4_DOC_NETWORKS for h in range(1, 255)]
    if field_type == "phone":
        return [f"+1-{a}-555-01{n:02d}" for a in NANP_AREA_CODES for n in range(100)]
    return None


def is_reserved_email(value: object) -> bool:
    """True when ``value`` is a syntactically valid address on a reserved
    (RFC 2606 / RFC 6761) domain."""
    if not isinstance(value, str) or value.count("@") != 1:
        return False
    local, domain = value.split("@")
    if not _LOCAL_RE.match(local) or ".." in local:
        return False
    domain = domain.lower()
    if not re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", domain):
        return False
    for suffix in RESERVED_EMAIL_SUFFIXES:
        if suffix.startswith("."):
            if domain.endswith(suffix):
                return True
        elif domain == suffix or domain.endswith("." + suffix):
            return True
    return False


_LOCAL_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*$")
