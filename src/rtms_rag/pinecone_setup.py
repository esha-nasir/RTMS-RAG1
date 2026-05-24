import os
try:
    # Newer SDK layout can expose classes via submodules.
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    from pinecone.pinecone import Pinecone
    from pinecone.db_control.models.serverless_spec import ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing env var: PINECONE_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "judgements")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "256"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

_pc = None
_index = None

def _extract_index_names(list_resp):
    if hasattr(list_resp, "names"):
        return [x for x in list_resp.names() if x]
    if isinstance(list_resp, list):
        return [
            i.get("name") if isinstance(i, dict) else getattr(i, "name", None)
            for i in list_resp
            if (i.get("name") if isinstance(i, dict) else getattr(i, "name", None))
        ]
    if isinstance(list_resp, dict):
        return [i.get("name") for i in list_resp.get("indexes", []) if i.get("name")]
    return []

def get_index():
    global _pc, _index
    if _index is not None:
        return _index

    _pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = _extract_index_names(_pc.list_indexes())

    if INDEX_NAME not in existing:
        _pc.create_index(
            name=INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    _index = _pc.Index(INDEX_NAME)
    return _index
