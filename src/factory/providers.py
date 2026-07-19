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
            parts = [p for p in name.lower().replace(".", " ").split() if p]
            local = ".".join(parts) if parts else self._slug()
        else:
            local = f"{self.first_name().lower()}.{self.last_name().lower()}"
        # Strip accents-ish characters that would be invalid in a local-part.
        local = "".join(c for c in local if c.isalnum() or c in "._-")
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    def phone(self) -> str:
        # Fictional +507 (Panama) style default; not a routable number range.
        area = self.rng.randint(200, 399)
        rest = self.rng.randint(1000, 9999)
        return f"+507-{area}-{rest}"

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
        # Documentation range 203.0.113.0/24 (RFC 5737) — safe, non-routable.
        return f"203.0.113.{self.rng.randint(1, 254)}"

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
