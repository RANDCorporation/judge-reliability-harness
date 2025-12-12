You are coordinating a sequence of concise edits to ensure the assistant conversation SATISFIES a specific RUBRIC CRITERION while staying coherent across turns.

Rubric Target ({rubric_id}):

```
{rubric_instructions}
```

Assistant Transcript (0-based indices, assistant turns only):

```
{assistant_transcript}
```

Plan a concise chain of at most {max_steps} edits that will bring the transcript into clear COMPLIANCE with the rubric criterion. The edits should be realistic, minimal, and coherent, building on preceding changes so the resulting story stays self-consistent.

Return only valid JSON with the following shape:

```json
{{
  "thesis": "<overall satisfaction narrative>",
  "steps": [
    {{
      "message_index": <int 0-based assistant index>,
      "goal": "<goal for this edit>",
      "rationale": "<how it advances satisfaction>"
    }}
  ]
}}
```

Guidelines:

- Target assistant turns only.
- Prefer concise additions, clarifications, checks, or validations that are strictly necessary to satisfy the rubric.
- Avoid meta statements about rules/policies; express the compliance naturally within the assistant’s voice.
- Honour chronological dependencies: later edits must respect changes introduced earlier.
- If fewer than {max_steps} edits are needed, return fewer steps.
- Produce "no plan" (null result) only when the transcript already satisfies the rubric with no edits.




