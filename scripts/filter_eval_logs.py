import os

from inspect_ai.log import read_eval_log, write_eval_log


def filter_log(input_path: str, keep_epochs: list[int]):
	"""
	Reads an eval log, filters samples by epoch, and writes to a new file.

	Args:
	    input_path: Path to the input .eval file.
	    keep_epochs: List of epoch numbers to keep.
	"""
	if not os.path.exists(input_path):
		print(f"Error: File not found: {input_path}")
		return

	print(f"Reading log: {input_path}")
	log = read_eval_log(input_path)

	initial_count = len(log.samples)
	print(f"  Initial samples: {initial_count}")

	# Filter samples
	# Assuming sample.epoch is the attribute.
	# Note: Inspect AI samples usually have an 'epoch' attribute if run with multiple epochs.
	# If 'epoch' is not present, we might need to check how it is stored.
	# Based on standard usage, it is typically log.samples[i].epoch

	filtered_samples = [s for s in log.samples if s.epoch in keep_epochs]

	log.samples = filtered_samples
	final_count = len(log.samples)
	print(f"  Filtered samples: {final_count} (kept epochs {keep_epochs})")

	# Create output path
	base, ext = os.path.splitext(input_path)
	output_path = f"{base}_filtered{ext}"

	print(f"Writing filtered log to: {output_path}")
	write_eval_log(log, output_path)
	print("-" * 40)


def main():
	base_dir = "inputs/data/stratus"

	files_to_process = [
		{"filename": "2025-12-05T00-32-25+00-00_evescape-task_cix6WPTbYcu9cp5vJpF4Qr.eval", "epochs": [4]},
		{"filename": "2025-12-06T02-44-03+00-00_evescape-task_3AETiXTm8PtU4dNzeRZuCR.eval", "epochs": [4]},
	]

	for item in files_to_process:
		input_path = os.path.join(base_dir, item["filename"])
		filter_log(input_path, item["epochs"])


if __name__ == "__main__":
	main()
