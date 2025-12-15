#!/usr/bin/env python
"""
Generate heatmap visualizations for judge performance across reliability tests.

This script reads JSON reports from a specified output directory and creates
a heatmap showing the performance (metric scores) of each judge/model across
different reliability tests. It also includes majority vote results if available.

Usage:
	python scripts/generate_judge_heatmap.py <output_directory>
	python scripts/generate_judge_heatmap.py outputs/fortress_20251119_1124
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_metric_value(value: Optional[str]) -> float:
	"""
	Convert metric value from string percentage to float.

	Args:
		value: Metric value as string (e.g., "75.00%") or None

	Returns:
		Float value between 0 and 1, or NaN if value is None
	"""
	if value is None or value == "None":
		return np.nan
	if isinstance(value, str) and value.endswith("%"):
		return float(value.rstrip("%")) / 100
	return float(value)


def load_reports(output_dir: Path) -> Dict[str, Dict]:
	"""
	Load all report JSON files from the output directory.

	Args:
		output_dir: Path to the output directory containing report files

	Returns:
		Dictionary mapping judge/model names to their report data
	"""
	reports = {}
	report_files = list(output_dir.glob("*_report.json"))

	if not report_files:
		raise ValueError(f"No report files found in {output_dir}")

	for report_file in report_files:
		with open(report_file, "r", encoding="utf-8") as f:
			report = json.load(f)

		# Extract judge/model name
		meta = report.get("meta_data", {})

		# Check if this is a majority vote report
		if "aggregation_method" in meta and meta["aggregation_method"] == "majority_vote":
			judge_name = "Majority Vote"
		elif "autograder_model" in meta:
			# Prefer full model name from autograder_model
			judge_name = meta["autograder_model"]
		elif "judge_name" in meta:
			# Fallback to short judge name
			judge_name = meta["judge_name"]
		else:
			# Final fallback to filename
			judge_name = report_file.stem.replace("_report", "")

		reports[judge_name] = report

	return reports


def extract_metrics_matrix(reports: Dict[str, Dict]) -> pd.DataFrame:
	"""
	Extract metrics from reports and organize into a matrix.

	Args:
		reports: Dictionary mapping judge names to report data

	Returns:
		DataFrame with tests as rows and judges as columns
	"""
	# Collect all test names and their sample sizes
	all_tests = set()
	test_sample_sizes = {}
	for report in reports.values():
		metrics = report.get("all_metrics", {})
		all_tests.update(metrics.keys())
		# Extract sample sizes for each test
		for test_name, test_data in metrics.items():
			if test_name not in test_sample_sizes:
				test_metrics = test_data.get("metrics", {})
				sample_size = test_metrics.get("total", "?")
				test_sample_sizes[test_name] = sample_size

	# Build matrix
	matrix_data = {}
	for judge_name, report in reports.items():
		metrics = report.get("all_metrics", {})
		judge_scores = {}

		for test_name in all_tests:
			# Create test label with sample size
			sample_size = test_sample_sizes.get(test_name, "?")
			test_label = f"{test_name} (N={sample_size})"

			if test_name in metrics:
				test_metrics = metrics[test_name].get("metrics", {})
				# Prefer metric_score, fallback to pass_rate
				score = test_metrics.get("metric_score") or test_metrics.get("pass_rate")
				judge_scores[test_label] = parse_metric_value(score)
			else:
				judge_scores[test_label] = np.nan

		matrix_data[judge_name] = judge_scores

	# Create DataFrame
	df = pd.DataFrame.from_dict(matrix_data, orient="columns")

	# Sort rows (tests) alphabetically
	df = df.sort_index()

	# Sort columns: put "Majority Vote" last if it exists
	columns = sorted([col for col in df.columns if col != "Majority Vote"])
	if "Majority Vote" in df.columns:
		columns.append("Majority Vote")
	df = df[columns]

	return df


def create_heatmap(
	df: pd.DataFrame,
	output_path: Path,
	title: str = "Judge Performance Across Reliability Tests",
	figsize: Optional[tuple] = None,
	cmap: str = "RdYlGn",
	vmin: float = 0.0,
	vmax: float = 1.0,
) -> None:
	"""
	Create and save a heatmap visualization.

	Args:
		df: DataFrame with tests as rows and judges as columns
		output_path: Path where to save the heatmap image
		title: Title for the heatmap
		figsize: Figure size (width, height), auto-calculated if None
		cmap: Colormap to use
		vmin: Minimum value for colormap
		vmax: Maximum value for colormap
	"""
	# Auto-calculate figure size if not provided
	if figsize is None:
		n_tests = len(df.index)
		n_judges = len(df.columns)
		width = max(8, n_judges * 1.5)
		height = max(6, n_tests * 0.5)
		figsize = (width, height)

	# Create figure
	plt.figure(figsize=figsize)

	# Create heatmap
	sns.heatmap(
		df,
		annot=True,
		fmt=".2%",
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		cbar_kws={"label": "Metric Score"},
		linewidths=0.5,
		linecolor="gray",
	)

	plt.title(title, fontsize=14, fontweight="bold", pad=20)
	plt.xlabel("Judge / Model", fontsize=12, fontweight="bold")
	plt.ylabel("Reliability Test", fontsize=12, fontweight="bold")
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.tight_layout()

	# Save figure
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()

	print(f"✅ Heatmap saved to {output_path}")


def generate_summary_stats(df: pd.DataFrame, output_path: Path) -> None:
	"""
	Generate and save summary statistics.

	Args:
		df: DataFrame with tests as rows and judges as columns
		output_path: Path where to save the summary statistics
	"""
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("# Judge Performance Summary Statistics\n\n")

		# Overall statistics by judge
		f.write("## Performance by Judge\n\n")
		f.write("| Judge | Mean Score | Std Dev | Min Score | Max Score |\n")
		f.write("|-------|------------|---------|-----------|----------|\n")

		for judge in df.columns:
			scores = df[judge].dropna()
			if len(scores) > 0:
				f.write(
					f"| {judge} | {scores.mean():.2%} | {scores.std():.2%} | "
					f"{scores.min():.2%} | {scores.max():.2%} |\n"
				)

		# Overall statistics by test
		f.write("\n## Performance by Test\n\n")
		f.write("| Test | Mean Score | Std Dev | Min Score | Max Score |\n")
		f.write("|------|------------|---------|-----------|----------|\n")

		for test in df.index:
			scores = df.loc[test].dropna()
			if len(scores) > 0:
				f.write(
					f"| {test} | {scores.mean():.2%} | {scores.std():.2%} | {scores.min():.2%} | {scores.max():.2%} |\n"
				)

	print(f"✅ Summary statistics saved to {output_path}")


def main():
	"""Main entry point for the script."""
	parser = argparse.ArgumentParser(
		description="Generate heatmap visualizations for judge performance across reliability tests."
	)
	parser.add_argument(
		"output_dir",
		type=str,
		help="Path to the output directory containing report JSON files",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Judge Performance Across Reliability Tests",
		help="Title for the heatmap (default: 'Judge Performance Across Reliability Tests')",
	)
	parser.add_argument(
		"--cmap",
		type=str,
		default="RdYlGn",
		help="Colormap to use (default: 'RdYlGn')",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default="judge_performance_heatmap.png",
		help="Output filename for the heatmap (default: 'judge_performance_heatmap.png')",
	)
	parser.add_argument(
		"--no-stats",
		action="store_true",
		help="Skip generating summary statistics file",
	)

	args = parser.parse_args()

	# Validate output directory
	output_dir = Path(args.output_dir)
	if not output_dir.exists():
		print(f"❌ Error: Output directory not found: {output_dir}")
		return 1

	if not output_dir.is_dir():
		print(f"❌ Error: Path is not a directory: {output_dir}")
		return 1

	try:
		# Load reports
		print(f"📂 Loading reports from {output_dir}...")
		reports = load_reports(output_dir)
		print(f"   Found {len(reports)} report(s)")

		# Extract metrics matrix
		print("📊 Extracting metrics...")
		df = extract_metrics_matrix(reports)
		print(f"   Tests: {len(df.index)}, Judges: {len(df.columns)}")

		# Create heatmap
		print("🎨 Creating heatmap...")
		heatmap_path = output_dir / args.output_name
		create_heatmap(df, heatmap_path, title=args.title, cmap=args.cmap)

		# Generate summary statistics
		if not args.no_stats:
			print("📈 Generating summary statistics...")
			stats_path = output_dir / "judge_performance_summary.md"
			generate_summary_stats(df, stats_path)

		print("\n✨ Done!")
		return 0

	except Exception as e:
		print(f"❌ Error: {e}")
		import traceback

		traceback.print_exc()
		return 1


if __name__ == "__main__":
	exit(main())
