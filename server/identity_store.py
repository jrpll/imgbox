import datetime
import hashlib
import io
import os
from pathlib import Path

import lancedb
import pyarrow as pa
from PIL import Image, ExifTags

_DATETIME_ORIGINAL = next((k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"), None)
_DATETIME = next((k for k, v in ExifTags.TAGS.items() if v == "DateTime"), None)


def _exif_taken_at(image_bytes: bytes) -> str | None:
    """Extract EXIF DateTimeOriginal (or DateTime) as ISO string, or None."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            exif = im.getexif()
            # DateTimeOriginal lives in the Exif sub-IFD (pointer tag 0x8769)
            sub = exif.get_ifd(0x8769)
            raw = sub.get(_DATETIME_ORIGINAL) or exif.get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
        if not raw:
            return None
        # EXIF format: "YYYY:MM:DD HH:MM:SS"
        dt = datetime.datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except Exception:
        return None

DATA_ROOT = Path(os.path.expanduser("~/.imgbox/identity"))
DB_PATH = DATA_ROOT / "db"
ORIGINALS = DATA_ROOT / "originals"
CROPS = DATA_ROOT / "crops"

for d in (DB_PATH, ORIGINALS, CROPS):
    d.mkdir(parents=True, exist_ok=True)

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 512)),
    pa.field("bbox", pa.list_(pa.float32(), 4)),
    pa.field("det_score", pa.float32()),
    pa.field("kps", pa.list_(pa.list_(pa.float32(), 2), 5)),
    pa.field("age", pa.int32()),
    pa.field("gender", pa.int32()),
    pa.field("source_filename", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("created_at", pa.string()),
    pa.field("model_name", pa.string()),
    pa.field("original_ext", pa.string()),
])

_db = lancedb.connect(str(DB_PATH))


def _table():
    try:
        return _db.open_table("identities")
    except (FileNotFoundError, ValueError):
        return _db.create_table("identities", schema=SCHEMA)


def _row_exists(table, id_: str) -> bool:
    try:
        return table.count_rows(filter=f"id = '{id_}'") > 0
    except Exception:
        return False


def insert(*, image_bytes: bytes, source_filename: str, model_name: str, embed_result: dict) -> tuple[str, bool]:
    """Insert (or repair) one identity row. Returns (id, was_new).

    Idempotent: checks each artifact (original, crop, DB row) independently and
    writes only what's missing. This recovers from interrupted previous runs
    where, e.g., the original was written but the crop or row was not.
    """
    id_ = hashlib.sha256(image_bytes).hexdigest()
    ext = Path(source_filename).suffix.lower() or ".jpg"
    original_p = ORIGINALS / f"{id_}{ext}"
    crop_p = CROPS / f"{id_}.jpg"
    table = _table()

    was_new = not (original_p.exists() and crop_p.exists() and _row_exists(table, id_))

    if not original_p.exists():
        original_p.write_bytes(image_bytes)

    if not crop_p.exists():
        crop = embed_result["crop"].copy()
        crop.thumbnail((512, 512))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=90)
        crop_p.write_bytes(buf.getvalue())

    if not _row_exists(table, id_):
        row = {
            "id": id_,
            "embedding": embed_result["embedding"],
            "bbox": embed_result["bbox"],
            "det_score": embed_result["det_score"],
            "kps": embed_result["kps"],
            "age": embed_result["age"],
            "gender": embed_result["gender"],
            "source_filename": source_filename,
            "width": embed_result["width"],
            "height": embed_result["height"],
            "created_at": _exif_taken_at(image_bytes) or "",
            "model_name": model_name,
            "original_ext": ext,
        }
        table.add([row])

    return id_, was_new


_LIGHT_COLS = [f.name for f in SCHEMA if f.name not in ("embedding", "kps")]


def list_all() -> list[dict]:
    """Return all rows, excluding heavy fields (embedding, kps)."""
    rows = _table().to_lance().to_table(columns=_LIGHT_COLS).to_pylist()
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def crop_path(id_: str) -> Path | None:
    p = CROPS / f"{id_}.jpg"
    return p if p.exists() else None


def original_path(id_: str) -> Path | None:
    for f in ORIGINALS.glob(f"{id_}.*"):
        return f
    return None


def find_similar(id_: str, k: int = 2) -> list[dict]:
    """Top-k nearest identities by embedding distance, excluding self.

    Returns rows with the same shape as list_all() plus a `_distance` field.
    """
    table = _table()
    arrow = table.to_lance().to_table(filter=f"id = '{id_}'", columns=["embedding"])
    if arrow.num_rows == 0:
        return []
    query_vec = arrow["embedding"][0].as_py()

    return (
        table.search(query_vec, vector_column_name="embedding")
        .where(f"id != '{id_}'")
        .select(_LIGHT_COLS)
        .limit(k)
        .to_arrow()
        .to_pylist()
    )


def delete(id_: str) -> bool:
    """Remove the DB row + original + crop. Returns True if anything existed."""
    table = _table()
    existed = _row_exists(table, id_)
    if existed:
        table.delete(f"id = '{id_}'")
    crop_p = CROPS / f"{id_}.jpg"
    if crop_p.exists():
        crop_p.unlink()
        existed = True
    for f in ORIGINALS.glob(f"{id_}.*"):
        f.unlink()
        existed = True
    return existed
