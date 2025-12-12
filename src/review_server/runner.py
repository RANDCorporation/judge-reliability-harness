from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List

import requests
import uvicorn

from core import console

from .server import ReviewItem, create_app


def _start_enter_watcher() -> threading.Event:
	"""
	Starts a background thread that sets an Event when Enter is pressed.
	Returns the Event object.
	"""
	enter_event = threading.Event()

	def wait_input():
		try:
			sys.stdin.readline()
			enter_event.set()
		except Exception:
			pass  # ignore stdin errors

	thread = threading.Thread(target=wait_input, daemon=True)
	thread.start()
	return enter_event


@contextmanager
def _uvicorn_server(app, host: str, port: int):
	config = uvicorn.Config(
		app,
		host=host,
		port=port,
		log_level="warning",
		access_log=False,
	)
	server = uvicorn.Server(config)

	import threading

	thread = threading.Thread(target=server.run, daemon=True)
	thread.start()

	# Wait until server is started
	timeout = time.time() + 10
	while not server.started and time.time() < timeout:
		time.sleep(0.05)

	try:
		yield server
	finally:
		server.should_exit = True
		thread.join(timeout=5)


class ReviewSession:
	"""Client helper that allows streaming items and collecting decisions."""

	def __init__(self, batch_id: str, url: str, initial_items: List[ReviewItem]):
		self.batch_id = batch_id
		self.url = url
		self._items: Dict[str, ReviewItem] = {item.sample_id: item for item in initial_items}

	def add_item(self, review_item: ReviewItem) -> None:
		self._items[review_item.sample_id] = review_item
		resp = requests.post(
			f"{self.url}/api/items",
			json=review_item.model_dump(),
			timeout=10,
		)
		resp.raise_for_status()

	def add_items(self, items: List[ReviewItem]) -> None:
		for item in items:
			self.add_item(item)

	def wait_for_finalize(self) -> str | None:
		"""Block until the review UI finalizes or the user presses Enter to continue."""
		try:
			enter_event = _start_enter_watcher()
			while True:
				try:
					resp = requests.get(f"{self.url}/api/status", timeout=5)
					resp.raise_for_status()
					data = resp.json()
					if data.get("finalized"):
						return "ui"
				except Exception:
					# The server may still be starting up or temporarily unavailable
					pass

				# Check if Enter has been pressed
				if enter_event.is_set():
					return "manual"

				time.sleep(0.5)
		except KeyboardInterrupt:
			console.print("Review interrupted via KeyboardInterrupt; continuing.")
			return "interrupt"

	def _fetch_decisions(self) -> List[Dict[str, Any]]:
		resp = requests.get(f"{self.url}/api/decisions", timeout=10)
		resp.raise_for_status()
		return resp.json()

	def collect_results(self) -> Dict[str, Any]:
		idx_to_item = {sid: item for sid, item in self._items.items()}
		decisions = self._fetch_decisions()
		accepted: List[Dict[str, Any]] = []
		rejected: List[Dict[str, Any]] = []
		edited: List[Dict[str, Any]] = []

		for decision in decisions:
			status = str(decision.get("status", "")).lower()
			sid = decision.get("sample_id")
			item = idx_to_item.get(sid)
			if not item:
				continue
			base = item.model_dump()
			if status == "accepted":
				if decision.get("edited_response") is not None:
					base["generated_response"] = decision["edited_response"]
				if decision.get("edited_expected") is not None:
					base["expected_label"] = decision["edited_expected"]
				accepted.append(base)
			elif status == "rejected":
				rejected.append(base)
			elif status == "edited":
				if decision.get("edited_response") is not None:
					base["generated_response"] = decision["edited_response"]
				if decision.get("edited_expected") is not None:
					base["expected_label"] = decision["edited_expected"]
				edited.append(base)

		return {
			"accepted": accepted,
			"rejected": rejected,
			"edited": edited,
		}


@contextmanager
def run_review_session(
	batch_id: str,
	items: List[Dict[str, Any]],
	output_dir: Path,
	host: str = "127.0.0.1",
	port: int = 8765,
	open_browser: bool = False,
) -> Generator[ReviewSession, None, None]:
	"""Launch a review session and yield a streaming client for interaction."""

	review_items = [ReviewItem(**it) for it in items]
	app = create_app(batch_id=batch_id, items=review_items)

	with _uvicorn_server(app, host=host, port=port):
		url = f"http://{host}:{port}"
		if open_browser:
			try:
				import webbrowser

				webbrowser.open(url)
			except Exception:
				pass

		print(f"🔎 JRH Review UI running at {url}")
		print("When finished, click Finalize in the UI or press Enter here.")

		session = ReviewSession(batch_id=batch_id, url=url, initial_items=review_items)
		try:
			yield session
		finally:
			pass
