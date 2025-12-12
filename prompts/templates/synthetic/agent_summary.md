You are tracking the evolving state of an agent conversation. Produce context that helps a downstream LLM reason about what has already happened.

Conversation transcript (chronological):
{chronological_log}

Return a compact JSON object with the following shape:
```
{{
  "summary": "<2-3 sentence high-level synopsis of progress so far>",
  "user_intent": "<one sentence describing the user's core objective>",
  "assistant_progress": ["<ordered bullet of key actions the assistant has taken so far>"],
  "outstanding_items": ["<pending follow-ups, open questions, or tasks that remain>"],
  "risks": ["<issues, blockers, or mistakes to monitor>"]
}}
```
Keep each string under 160 characters. Omit list entries if nothing fits. Do not include any extra keys or text outside the JSON.
