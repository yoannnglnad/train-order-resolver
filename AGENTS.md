# Repository Guidelines

## Project Structure & Module Organization
- Core docs live at `README.md`, `prd.md`, and `project.pdf`; keep them updated when interfaces or workflows change.
- `_bmad/` holds internal BMAD agent/workflow configs; `_bmad-output/` stores generated notes—do not overwrite manually.
- Source code is expected under `src/` (e.g., `src/nlp`, `src/pathfinding`, `src/utils`), with datasets and models in `data/` (`data/raw`, `data/dataset`, `data/models`), tests in `tests/`, and the CLI entry point in `main.py`. Mirror this layout when adding new modules.

## Build, Test, and Development Commands
- Python 3.10+ with a virtual env: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies once `requirements.txt` is present: `python -m pip install -r requirements.txt`.
- Run the CLI on a file: `python main.py --input inputs.txt`; stream from stdin via `echo "1,Je veux aller à Lille depuis Marseille" | python main.py`.
- Generate synthetic data (expected): `python src/nlp/dataset_generator.py --count 10000 --output data/dataset/train.csv`; train the NER model: `python src/nlp/train.py --epochs 3`.
- Execute the suite: `python -m pytest`; add `-k <pattern>` for focused runs.

## Coding Style & Naming Conventions
- Follow PEP 8; prefer `black --line-length 100` for formatting and `ruff` for linting. Include type hints on public functions.
- Packages and modules: `lower_snake_case`; classes: `PascalCase`; functions/variables: `snake_case`; constants: `UPPER_SNAKE_CASE`.
- Keep CLI flags long-form with hyphens (e.g., `--input-file`), and centralize configuration defaults in a single module under `src/utils/`.

## Testing Guidelines
- Use `pytest` with files named `test_<module>.py` under `tests/`, mirroring the source structure.
- Provide fixtures in `tests/fixtures` for small CSV graphs and sample sentences; cover both valid trips and `INVALID` paths.
- Aim for high coverage on parsing, graph construction, and itinerary generation; include smoke tests for the CLI pipeline using short input files.

## Commit & Pull Request Guidelines
- Git history is empty; adopt Conventional Commits (e.g., `feat: add pathfinding graph builder`, `fix: handle unknown departure station`).
- Branch naming: `feature/<scope>`, `fix/<scope>`, or `chore/<scope>`.
- PRs should state the problem, the solution, commands run (tests/training), and note any data/model artifacts excluded from git. Update `README.md` or `prd.md` when changing interfaces or assumptions.

## Security & Data Handling
- Do not commit large SNCF CSVs or trained models; keep them in `data/` and add to `.gitignore` as needed.
- Avoid secrets in code or notebooks; prefer environment variables and document required keys in the README.
- Validate external data sources before training to prevent corrupt graph nodes or mislabeled entities.
