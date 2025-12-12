# src/synthetic_data_pipeline/stochastic_stability.py

import numpy as np
import pandas as pd

from core import console
from schemas import TestStochasticStabilityConfig
from .utils import get_sample


def generate_stochastic_stability(
	config: TestStochasticStabilityConfig,
	df: pd.DataFrame,
) -> pd.DataFrame:
	"""
	Generate duplicated df with different seeds in order to measure consistency.

	Args:
	    config (TestStochasticStabilityConfig): Config to generate perturbations for stochastic stability test.
	    df (pd.DataFrame): Original dataset.

	Returns:
	    pd.DataFrame: The new dataframe with duplicated entries.
	"""
	if df.empty:
		console.print("[bold red]Original dataset is empty. Skipping test.[/]")
		return pd.DataFrame()

	sample_df = get_sample(df, config.sample_num_from_orig)

	# Set seed and generate base seeds
	np.random.seed(config.seed)
	base_seeds = np.random.randint(0, 2**31 - 1, size=config.number_of_seeds)

	# Repeat the DataFrame for each seed and retry
	repeated_df = pd.concat([sample_df] * config.number_of_seeds * config.repetitions, ignore_index=True)

	# Generate seed and retry_idx columns
	repeated_df["seed"] = np.repeat(base_seeds, config.repetitions * len(sample_df))
	repeated_df["retry_idx"] = np.tile(np.repeat(np.arange(config.repetitions), len(sample_df)), config.number_of_seeds)

	return repeated_df
