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
