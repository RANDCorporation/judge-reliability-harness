This is a guide for using the Judge Reliability Harness with a simple example.

# Setup Virtual Environment, Clone Project, and Install Dependencies

Clone project with:

- `git clone https://github.com/tasp-evals/judge-reliability-harness.git`

Move into the project:

- `cd judge-reliability-harness`

Setup virtual environment with:

- `python -m venv venv`
- Activate with:
  - Linux/macOS: `source venv/bin/activate`
  - On Windows (Command Prompt): `venv\Scripts\activate`
  - On Windows (PowerShell): `venv\Scripts\Activate.ps1`

Install dependencies:

* ```uv sync --extra dev --native-tls```

Set environment variables:
* Linux/macOS: ```echo -e "OPENAI_API_KEY=your_api_key_here\nOPENAI_ORG_ID=your_org_id_here" > .env```
* On Windows (PowerSHell): 
```
"OPENAI_API_KEY=your_api_key_here" | Out-File -Encoding utf8 .env
"OPENAI_ORG_ID=your_org_id_here" | Out-File -Encoding utf8 -Append .env
```

# Deploy JRH

## Deploy with default options

- This runs the _binary harmbench_ program, by default: `python -m main`
- This runs the _persuade_ program: `python -m main ./inputs/configs/config_persuade.yml`

See default options here: `./src/configs/default_config.yml `

## Deploy with custom options

Step 1: Upload new data set for a project named "PROJECT_NAME"

- Place new data set into `./inputs/data/PROJECT_NAME/data.csv`
- Put new rubric into `./inputs/data/PROJECT_NAME/rubric.txt`
- Put new instructions into `./inputs/data/PROJECT_NAME/instructions.txt`

Step 2: Create new config file

- `cp ./src/configs/default_config.yml ./inputs/configs/custom_config.yml`
- Change YAML file with desired configuration. Note that the fields that are in config_persuade.yml are REQUIRED.

Step 3: Run with custom options

- `python -m main ./inputs/configs/custom_config.yml`

The data will appear in the output_dir specified by your YAML file. By default, this is:

- Results: `./outputs/PROJECT_NAME_TIMESTAMP/report.md`
- Perturbations: `./outputs/PROJECT_NAME_TIMESTAMP/TEST_NAME_perturbations.csv`

## Running Unit Tests

To run all the unit tests in the project, use the following command from the project root:

```bash
pytest
```

## Linting & formatting

We use ruff as our linter. Before committing code, run these commands and fix any linter issues:

```bash
ruff format
ruff check --fix
```

# Sunset the setup

Deactivate virtual environment with `deactivate`
