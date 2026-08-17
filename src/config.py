import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

NHTSA_COMPLAINTS_DIR = DATA_DIR / "NHTSA Customer Complaints"
CRAIGSLIST_VEHICLES_CSV = DATA_DIR / "Used Cars Dataset" / "vehicles.csv"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHROMA_COLLECTION_NAME = "carscout"
