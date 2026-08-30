import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

NHTSA_COMPLAINTS_DIR = DATA_DIR / "NHTSA Customer Complaints"
NHTSA_RECALLS_CSV = NHTSA_COMPLAINTS_DIR / "recalls.csv"
CRAIGSLIST_VEHICLES_CSV = DATA_DIR / "Used Cars Dataset" / "vehicles.csv"

DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"

PINECONE_INDEX_NAME = "carscout-complaints"

# (make, model, year, year_tolerance); year=None matches any model year.
# Shared across NHTSA complaint/recall ingestion and Craigslist price
# ingestion so all three datasets stay filtered to the same vehicles.
VEHICLE_SHORTLIST = [
    ("hyundai", "kona", 2020, 1),
    ("mazda", "mazda3", 2017, 1),
    ("kia", "forte", 2021, 1),
    ("hyundai", "elantra", 2021, 1),
    ("honda", "civic", 2015, 1),
    ("toyota", "corolla", None, None),
]

# One real listing per shortlist vehicle, sourced from the same Craigslist
# dataset used for price comps (data/Used Cars Dataset/vehicles.csv) - real
# VIN, real price/odometer/condition from that row. Each VIN was verified to
# decode cleanly via NHTSA's vPIC API (make/model/year match) before being
# added here. The demo UI lets the user pick one instead of typing vehicle/
# price/odometer/condition by hand - only the symptom stays free text, since
# that's the buyer's own concern, not something a listing states.
VIN_DEMO_LISTINGS = [
    {
        "vin": "KM8K2CAA8LU462861",
        "make": "Hyundai", "model": "Kona", "year": 2020,
        "price": 24513, "odometer": 6513, "condition": "like new",
    },
    {
        "vin": "3MZBN1L3XHM151293",
        "make": "Mazda", "model": "Mazda3", "year": 2017,
        "price": 17433, "odometer": 35564, "condition": None,
    },
    {
        "vin": "3KPF24AD1LE141393",
        "make": "Kia", "model": "Forte", "year": 2020,
        "price": 18200, "odometer": 25743, "condition": "like new",
    },
    {
        "vin": "KMHD74LF1LU900547",
        "make": "Hyundai", "model": "Elantra", "year": 2020,
        "price": 14499, "odometer": 13145, "condition": "excellent",
    },
    {
        "vin": "2HGFG3B06FH517149",
        "make": "Honda", "model": "Civic", "year": 2015,
        "price": 15729, "odometer": 59300, "condition": "excellent",
    },
    {
        "vin": "5YFBURHE8HP655686",
        "make": "Toyota", "model": "Corolla", "year": 2017,
        "price": 12500, "odometer": 33591, "condition": "good",
    },
]
