### [Read the docs](https://randcorporation.github.io/judge-reliability-harness/)


## Project Overview
The Judge Reliability Harness (JRH) orchestrates end-to-end evaluations of automated judges. It standardizes how datasets are prepared, perturbations are generated, and model outputs are rescored so teams can compare judge reliability across tasks. The harness coordinates data ingestion, perturbation pipelines, automated grading, and reporting into reproducible runs.

## Why JRH?
LLMs are increasingly used as judges or to score, rank, or classify AI outputs in AI evaluations. Human evaluation yields high quality judgments but is expensive and difficult to scale, which has motivated the widespread use of LLMs as judges in place of human annotators However, the reliability of judge system comfiguration, including the LLM judge model, rubric, and prompt templates, are rarely evaluated and measured in a systematic manner or reported alongside benchmark evaluation results. Point estimates of agreement with human raters on small validation sets provide limited assurance about how a judge will respond to realistic variations in inputs, such as changes in formatting, paraphrasing, verbosity, or sampling parameters. This gap between the central role of LLM judges and the limited tools available to characterize their reliability makes it difficult for practitioners and decision makers to understand how much confidence to place in AI evaluation results.

We introduce the Judge Reliability Harness (JRH), an open source library that generates validation suites for any LLM judge on both agentic and free-response benchmarks. JRH generates reliability tests that measure grading accuracy via label flipped responses, invariance to formatting and paraphrasing, susceptibility to verbosity bias, stochastic stability under repeated sampling, and calibration across an ordinal grading scale. JRH features a human-in-the-loop review process for generated reliability tests through a user interface that gives full control to accept, reject, or edit the tests. Across a range of candidate judges, it aggregates pass rates, confidence intervals, and cost curves into standardized reports. By making reliability testing configurable, reproducible, and inexpensive, JRH aims to support a more transparent and trustworthy use of LLM judges in both research and deployment contexts.

# Setup Virtual Environment, Clone Project, and Install Dependencies

Clone project with:

- `git clone https://github.com/RANDCorporation/judge-reliability-harness.git`

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
