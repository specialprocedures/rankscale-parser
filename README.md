# rankscale-parser

Parses a [Rankscale](https://rankscale.ai) export (UTF-16, tab-separated) into four CSVs:
`queries.csv`, `responses.csv`, `citations.csv` and `domains.csv`.

## Install

With [uv](https://docs.astral.sh/uv/) (`pacman -S uv`):

```bash
uv tool install git+https://github.com/specialprocedures/rankscale-parser.git
```

Or with pipx (`pacman -S python-pipx`):

```bash
pipx install git+https://github.com/specialprocedures/rankscale-parser.git
```

## Usage

```bash
rankscale-parser -i export.csv -o out/
```

- `-i, --input` — Rankscale export to parse (required)
- `-o, --output-dir` — where to write the CSVs (default: current directory; created if missing)

## Upgrade

```bash
uv tool upgrade rankscale-parser    # or: pipx upgrade rankscale-parser
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e .
```
