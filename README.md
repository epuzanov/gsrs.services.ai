# gsrs-services-ai

`gsrs-services-ai` provides `gsrs.services.ai` utilities for transforming GSRS model payloads into embedding-friendly chunks.

## Features

- Installable Python package exposing `gsrs.services.ai`
- `SubstanceChunker` for summary and section-level chunk generation
- Structured chunk metadata suitable for retrieval and ranking workflows
- Unit tests covering imports and chunk behavior

## Installation

Install from source:

```bash
pip install .
```

For local development:

```bash
pip install -e .
```

## Quick Usage

```python
from gsrs.model import Substance
from gsrs.services.ai import SubstanceChunker

payload = {
    "substanceClass": "concept",
    "uuid": "11111111-1111-1111-1111-111111111111",
    "names": [{"name": "Example Concept", "type": "cn", "languages": ["en"]}],
    "references": [{"docType": "SYSTEM"}],
    "_self": "https://example.test/gsrs",
    "version": "1",
}

substance = Substance.model_validate(payload)
chunks = SubstanceChunker().chunk(substance)
print(chunks[0]["section"])
print(chunks[0]["text"])
```

See [docs/usage.md](docs/usage.md) for more examples.

## Development

Run tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Recommended local workflow uses a project `.venv`; details are in [docs/development.md](docs/development.md).

## License

MIT. See [LICENSE](LICENSE).
