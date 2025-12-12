# main.py

"""
Main run file.

Run with:
> python -m main
"""

import typer
from pathlib import Path
import yaml

from src import Harness, SyntheticDataAdapter, save_config
from core import console, get_validated_admin_config
from synthetic_data_pipeline import build_registry


app = typer.Typer()


@app.command()
def run(
	config_path: Path = typer.Argument(
		Path("./src/configs/default_config.yml"), exists=True, help="Path to the configuration YAML file."
	),
):
	"""
	Runs the Judge Reliability Harness using a specified configuration file.
	"""
	console.print("\n[bold green] Starting Judge Reliability Harness...[/bold green]")
	console.print(f"Loading configuration from: [cyan]{config_path}[/cyan]")

	try:
		with open(config_path, "r") as f:
			config = yaml.safe_load(f)

		admin_config, merged_config = get_validated_admin_config(config)

		save_config(admin_config.output_dir, config_path.name, merged_config)

		validated_test_list = build_registry(admin_config, merged_config)
		synthetic_data_service = SyntheticDataAdapter(admin_config.perturbation_config, validated_test_list)
		harness = Harness(admin_config, synthetic_data_service)
		harness.run()

		console.print("\n[bold green]Harness execution finished successfully![/bold green]")

	except Exception as e:
		console.print(f"\n[bold red] An error occurred:[/bold red] {e}")
		import traceback

		traceback.print_exc()
		raise typer.Exit(code=1)


def main():
	app()


if __name__ == "__main__":
	main()
