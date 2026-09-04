"""ACE event-role schemas used by the standalone VPTD-EAE package."""

from __future__ import annotations


EVENT_SCHEMAS = {
    "LIFE||DIE": ("Agent", "Instrument", "Victim", "Place"),
    "MOVEMENT||TRANSPORT": ("Agent", "Artifact", "Vehicle", "Destination", "Origin"),
    "TRANSACTION||TRANSFER|MONEY": ("Giver", "Recipient", "Money"),
    "CONFLICT||ATTACK": ("Instrument", "Place", "Attacker", "Target"),
    "CONFLICT||DEMONSTRATE": ("Entity", "Police", "Instrument", "Place"),
    "CONTACT||MEET": ("Participant", "Place"),
    "CONTACT||PHONE|WRITE": ("Entity", "Instrument", "Place"),
    "JUSTICE||ARREST|JAIL": ("Agent", "Person", "Instrument", "Place"),
}


IMAGE_TO_ACE_ROLES = {
    "LIFE||DIE": {"victim": "Victim", "agent": "Agent", "instrument": "Instrument", "place": "Place"},
    "MOVEMENT||TRANSPORT": {
        "agent": "Agent",
        "person": "Artifact",
        "artifact": "Artifact",
        "instrument": "Instrument",
        "vehicle": "Vehicle",
        "destination": "Destination",
        "origin": "Origin",
    },
    "TRANSACTION||TRANSFER|MONEY": {
        "giver": "Giver",
        "recipient": "Recipient",
        "beneficiary": "Beneficiary",
        "money": "Money",
        "place": "Place",
    },
    "CONFLICT||ATTACK": {
        "attacker": "Attacker",
        "agent": "Attacker",
        "instrument": "Instrument",
        "place": "Place",
        "target": "Target",
        "victim": "Target",
    },
    "CONFLICT||DEMONSTRATE": {
        "police": "Police",
        "instrument": "Instrument",
        "demonstrator": "Entity",
        "participant": "Entity",
        "place": "Place",
    },
    "CONTACT||MEET": {"participant": "Entity", "place": "Place"},
    "CONTACT||PHONE|WRITE": {"participant": "Entity", "instrument": "Instrument", "place": "Place"},
    "JUSTICE||ARREST|JAIL": {
        "agent": "Agent",
        "person": "Person",
        "instrument": "Instrument",
        "place": "Place",
    },
}


def normalize_event_type(event_type: str) -> str:
    return event_type.replace(".", "||").replace(":", "||").replace("-", "|").upper()


def roles_for_event_type(event_type: str) -> tuple[str, ...]:
    return EVENT_SCHEMAS.get(normalize_event_type(event_type), ())


def map_visual_role(event_type: str, role: str) -> str:
    mapping = IMAGE_TO_ACE_ROLES.get(normalize_event_type(event_type), {})
    return mapping.get(role.strip().casefold(), role.strip())
