# src/core/tuils.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable, Any, Iterator, Tuple, Dict
import pandas as pd

from .constants import console


def threaded_executor(
	dataset: pd.DataFrame, worker_fn: Callable[[Dict[str, Any]], Any], max_workers: int
) -> Iterator[Tuple[Dict[str, Any], Any]]:
	"""
	Execute a worker function on each row of a dataset using threads and yield results as they complete.

	Args:
	    dataset (pandas.DataFrame): The dataset to process. Each row is passed to worker_fn as a dict.
	    worker_fn (Callable[[dict], Any]): Function to apply to each row. Should accept a row dictionary.
	    max_workers (int): Maximum number of threads to use.

	Yields:
	    tuple: A tuple `(row, result)` where:
	        - row (dict): The original row dictionary from the dataset.
	        - result (Any): The result returned by worker_fn for that row.

	Notes:
	    Exceptions raised by worker_fn are caught and printed; processing continues for other rows.
	    Results may be yielded in a different order than the original dataset.
	"""
	rows = dataset.to_dict(orient="records")
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = {executor.submit(worker_fn, row): row for row in rows}
		try:
			for future in tqdm(as_completed(futures), total=len(futures)):
				row = futures[future]
				try:
					yield row, future.result()
				except Exception as e:
					console.print(f"Error processing row {row}: {e}")
		except KeyboardInterrupt:
			console.print("[red]KeyboardInterrupt detected, shutting down threads...[/red]")
			# Cancel all pending futures
			for future in futures:
				future.cancel()
			executor.shutdown(wait=False)  # don't wait for threads to finish
			raise
