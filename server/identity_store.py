import datetime
import hashlib
import io
import os
from pathlib import Path

import lancedb
import pyarrow as pa

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
            "created_at": datetime.datetime.utcnow().isoformat(),
            "model_name": model_name,
            "original_ext": ext,
        }
        table.add([row])

    return id_, was_new


def list_all() -> list[dict]:
    """Return all rows, excluding heavy fields (embedding, kps)."""
    table = _table()
    df = table.to_pandas()
    if df.empty:
        return []
    cols = [c for c in df.columns if c not in ("embedding", "kps")]
    out = []
    for row in df[cols].to_dict(orient="records"):
        out.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()})
    out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return out


def crop_path(id_: str) -> Path | None:
    p = CROPS / f"{id_}.jpg"
    return p if p.exists() else None


def original_path(id_: str) -> Path | None:
    for f in ORIGINALS.glob(f"{id_}.*"):
        return f
    return None
