# Documentation website (MkDocs)

This repo can be browsed as a static documentation site using **MkDocs** with the **Read the Docs** theme.

MkDocs is configured under `docs/mkdocs/` and is set up to browse markdown across the repository (README files under `kernels/`, `tests/`, etc.).

## Prerequisites

- Python 3.8+

## Install (recommended: virtual environment)

```bash
python3 -m venv .venv-mkdocs
source .venv-mkdocs/bin/activate
python -m pip install --upgrade pip
python -m pip install -r docs/mkdocs/requirements.txt
```

## Serve locally (live reload)

```bash
python -m mkdocs serve -f docs/mkdocs/mkdocs.yml
```

Then open `http://127.0.0.1:8000/`.

## Build static site

```bash
python -m mkdocs build -f docs/mkdocs/mkdocs.yml
```

The output is written to `site/` (or a custom directory if you pass `-d`).

## Build with CMake

You can build the static site as part of a CMake build:

```bash
cmake -S docs -B build/docs
cmake --build build/docs --target pto_docs
```

The site is generated under `build/docs/site/`.

To serve locally:

```bash
cmake --build build/docs --target pto_docs_serve
```

## Notes

- The MkDocs source directory is `docs/mkdocs/src/`.
- `docs/mkdocs/gen_pages.py` publishes only curated canonical sources from `docs/isa/`, `docs/assembly/`, and getting-started pages; generated virtual pages should not be committed under `docs/mkdocs/src/`.
