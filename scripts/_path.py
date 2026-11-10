"""Put the project root on sys.path so the scripts can `import tada`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
