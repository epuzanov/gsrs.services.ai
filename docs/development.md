# Development

## Local Setup (.venv)

From the project root:

```bash
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run build smoke test:

```bash
python -m pip install build
python -m build
```

## Dependency Notes

- Runtime model dependency: `gsrs-model>=0.3.0`
- Runtime validation dependency: `pydantic>=2,<3`

## GitHub Publishing Checklist

1. Create the GitHub repository and add this project as the remote.
2. Commit current files and push the default branch.
3. Verify tests pass in CI or locally from `.venv`.
4. Create an annotated version tag (for example `v0.1.0`).
5. Push the tag and publish release notes on GitHub.
