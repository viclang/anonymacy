from pathlib import Path
from typing import Any
import srsly

def read_pickle(path: Path) -> Any:
    """Load object from a pickle file using srsly."""
    with path.open("rb") as f:
        return srsly.pickle_loads(f.read())


def write_pickle(path: Path,obj: Any) -> None:
    """Save object to a pickle file using srsly."""
    with path.open("wb") as f:
        f.write(srsly.pickle_dumps(obj))