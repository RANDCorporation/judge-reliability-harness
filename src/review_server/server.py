from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from schemas import ReviewItem, ReviewDecision


def create_app(batch_id: str, items: List[ReviewItem]):
	"""Create a FastAPI app for a single review batch and hold state in memory."""
	app = FastAPI(title="JRH Review UI", version="0.1.0")

	state = {
		"batch_id": batch_id,
		"items": {item.sample_id: item for item in items},
		"order": [item.sample_id for item in items],
		"decisions": {},
		"finalized": False,
		"connections": set(),
		"lock": asyncio.Lock(),
	}

	def _serialize_item(item: ReviewItem) -> Dict[str, Any]:
		data = item.model_dump()
		decision = state["decisions"].get(item.sample_id)
		if decision:
			data["decision"] = decision.model_dump()
			# Apply edited values whenever they exist so the UI reflects pending changes
			if decision.edited_response is not None:
				data["generated_response"] = decision.edited_response
			if decision.edited_expected is not None:
				data["expected_label"] = decision.edited_expected
		return data

	async def _broadcast(event: str, payload: Dict[str, Any]) -> None:
		stale = []
		for connection in list(state["connections"]):
			try:
				await connection.send_json({"type": event, "payload": payload})
			except Exception:
				stale.append(connection)
		for connection in stale:
			try:
				await connection.close()
			except Exception:
				pass
			state["connections"].discard(connection)

	@app.get("/api/batch")
	def get_batch():
		"""Return batch id and item count for the current session."""
		return {"batch_id": state["batch_id"], "count": len(state["items"])}

	@app.get("/api/items")
	async def list_items():
		"""Return all items in the current batch as JSON."""
		return [_serialize_item(state["items"][sid]) for sid in state["order"]]

	@app.get("/api/items/{sample_id}")
	async def get_item(sample_id: str):
		"""Return a single item by id or 404 if not present."""
		item = state["items"].get(sample_id)
		if not item:
			raise HTTPException(status_code=404, detail="Item not found")
		return _serialize_item(item)

	@app.get("/api/decisions")
	async def list_decisions():
		"""Return all recorded decisions for this batch."""
		return [decision.model_dump() for decision in state["decisions"].values()]

	@app.get("/api/status")
	async def session_status():
		"""Return current session metadata including finalize state."""
		return {
			"ok": True,
			"batch_id": state["batch_id"],
			"num_items": len(state["items"]),
			"num_decisions": len(state["decisions"]),
			"finalized": state["finalized"],
		}

	@app.post("/api/decisions")
	async def submit_decision(decision: ReviewDecision):
		"""Insert or update a decision for a given sample id."""
		if decision.sample_id not in state["items"]:
			raise HTTPException(status_code=404, detail="Item not found")
		state["decisions"][decision.sample_id] = decision
		payload = _serialize_item(state["items"][decision.sample_id])
		await _broadcast("decision_updated", payload)
		return {"ok": True}

	@app.get("/api/finalize")
	async def finalize():
		"""Signal that the batch is complete; the runner will proceed."""
		# Read decisions, stop server
		state["finalized"] = True
		await _broadcast(
			"finalized",
			{
				"batch_id": state["batch_id"],
				"num_decisions": len(state["decisions"]),
			},
		)
		return {
			"ok": True,
			"batch_id": state["batch_id"],
			"num_decisions": len(state["decisions"]),
			"finalized": True,
		}

	@app.post("/api/items")
	async def upsert_item(item: ReviewItem):
		"""Add a new item or update an existing one, then broadcast it."""
		async with state["lock"]:
			existed = item.sample_id in state["items"]
			state["items"][item.sample_id] = item
			if not existed:
				state["order"].append(item.sample_id)
		payload = _serialize_item(item)
		await _broadcast("item_updated" if existed else "item_created", payload)
		return {"ok": True, "status": "updated" if existed else "created"}

	@app.websocket("/ws/items")
	async def items_websocket(websocket: WebSocket):
		await websocket.accept()
		state["connections"].add(websocket)
		try:
			await websocket.send_json(
				{
					"type": "init",
					"payload": {
						"batch_id": state["batch_id"],
						"items": [_serialize_item(state["items"][sid]) for sid in state["order"]],
						"finalized": state["finalized"],
					},
				}
			)
			while True:
				try:
					await websocket.receive_text()
				except WebSocketDisconnect:
					break
		finally:
			state["connections"].discard(websocket)

	@app.get("/", response_class=HTMLResponse)
	async def index_page():
		html_path = Path("./src/review_server/index.html")
		return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

	return app
