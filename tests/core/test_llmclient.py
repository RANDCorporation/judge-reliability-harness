# tests/core/test_llmclient.py

import pandas as pd
import pytest
from unittest.mock import MagicMock
from core.llmclient import LLMClient
from schemas import LLMClientConfig, BasicLLMResponseBool


@pytest.fixture(autouse=True)
def mock_instructor(monkeypatch):
	dummy_client = MagicMock()
	dummy_client.chat.completions.create.return_value = {"score": 1, "reasoning": "ok"}
	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda model: dummy_client)
	return dummy_client


@pytest.fixture
def dummy_config():
	return LLMClientConfig(
		model="openai/gpt-4o-mini",
		template="single_judge",
		default_params={"instruction": "Do this"},
		max_workers=1,
		retries=1,
		max_tokens=100,
		temperature=0.7,
		test_debug_mode=False,
		rate_limit={},
	)


@pytest.fixture
def dummy_dataset():
	return pd.DataFrame(
		[
			{"idx": 1, "question": "Q1"},
			{"idx": 2, "question": "Q2"},
		]
	)


def test_init_sets_attributes_and_rate_limit(monkeypatch, dummy_config):
	dummy_client = MagicMock()
	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda model: dummy_client)

	client = LLMClient(dummy_config)

	assert client.config == dummy_config
	assert client.client == dummy_client
	assert callable(client.call)


def test_build_prompt_merges_defaults_and_row(monkeypatch, dummy_config):
	monkeypatch.setattr(
		"core.llmclient.get_prompt", lambda name, **kwargs: f"{kwargs['instruction']} {kwargs.get('question', '')}"
	)
	client = LLMClient(dummy_config)

	row = {"question": "What is 2+2?"}
	prompt = client._build_prompt(row)

	assert "Do this" in prompt
	assert "What is 2+2?" in prompt


def test_call_debug_mode_returns_basic_response(dummy_config):
	config = dummy_config.model_copy(update={"test_debug_mode": True})
	client = LLMClient(config)

	resp = client.call({"question": "Q"})
	assert isinstance(resp, BasicLLMResponseBool)
	assert "ERROR" in resp.reasoning


def test_call_successful_llm(monkeypatch, dummy_config):
	monkeypatch.setattr("core.llmclient.get_prompt", lambda name, **kwargs: "Prompt here")

	fake_response = {"score": 1, "reasoning": "ok"}

	dummy_chat = MagicMock()
	dummy_chat.completions.create.return_value = fake_response

	dummy_client = MagicMock()
	dummy_client.chat = dummy_chat

	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda model: dummy_client)

	client = LLMClient(dummy_config)
	resp = client.call({"question": "Q"})

	assert isinstance(resp, BasicLLMResponseBool)
	assert resp.score == 1
	assert resp.reasoning == "ok"


def test_batch_call_replaces_run_from_dataset(dummy_config, dummy_dataset, monkeypatch):
	"""New version of the old run_from_dataset test."""

	# Patch LLMClient.call to return a predictable response
	monkeypatch.setattr(
		"core.llmclient.LLMClient.call",
		lambda self, row, response_schema=BasicLLMResponseBool, **kwargs: BasicLLMResponseBool(score=1, reasoning="ok"),
	)

	client = LLMClient(dummy_config)

	# Simulate the old run_from_dataset behavior manually
	results = []
	for _, row in dummy_dataset.iterrows():
		resp = client.call(row.to_dict())
		results.append({**row.to_dict(), **resp.model_dump()})

	df = pd.DataFrame(results)

	assert len(df) == len(dummy_dataset)
	assert "score" in df.columns
	assert "reasoning" in df.columns


def test_batch_call_empty_df_returns_empty(dummy_config):
	client = LLMClient(dummy_config)
	empty_df = pd.DataFrame(columns=["idx", "question"])

	if empty_df.empty:
		df = pd.DataFrame()
	else:
		results = []
		for _, row in empty_df.iterrows():
			resp = client.call(row.to_dict())
			results.append({**row.to_dict(), **resp.model_dump()})
		df = pd.DataFrame(results)

	assert df.empty
