# Usage

This package exposes `SubstanceChunker` from `gsrs.services.ai` for converting GSRS objects into chunk dictionaries.

## Basic Chunking

```python
from gsrs.model import Substance
from gsrs.services.ai import SubstanceChunker

substance = Substance.model_validate(
    {
        "substanceClass": "concept",
        "uuid": "11111111-1111-1111-1111-111111111111",
        "names": [{"name": "Example Concept", "type": "cn", "languages": ["en"]}],
        "references": [{"docType": "SYSTEM", "citation": "generated"}],
        "_self": "https://example.test/gsrs",
        "version": "1",
    }
)

chunks = SubstanceChunker().chunk(substance)
```

Typical chunk keys:

- `chunk_id`: stable id for the chunk
- `document_id`: root GSRS substance uuid
- `source_url`: canonical source link when available
- `section`: logical section (`summary`, `names`, `codes`, `references`, ...)
- `text`: natural-language chunk content
- `metadata`: structured attributes for filtering and ranking

## Custom Chunk Wrapper

You can cast chunk outputs to another mapping-like class:

```python
class ChunkEnvelope(dict):
    pass

chunker = SubstanceChunker(**{"class": ChunkEnvelope})
wrapped = chunker.chunk(substance)
assert isinstance(wrapped[0], ChunkEnvelope)
```

## Import Surface

Use:

```python
from gsrs.services.ai import SubstanceChunker
```
