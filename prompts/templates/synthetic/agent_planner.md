You are coordinating a sequence of edits to degrade a specific RUBRIC CRITERIA in an assistant conversation while staying coherent across turns.

Rubric Target ({rubric_id}):

```
{rubric_instructions}
```

Assistant Transcript (0-based indices, assistant turns only):

```
{assistant_transcript}
```

Plan a chain of at most {max_steps} edits that will cause the transcript to fail the rubric criteria. The edits should be realistic and coherent, building on preceding changes so the resulting story stays self-consistent.

Return **only** valid JSON with the following shape:

```json
{{
  "thesis": "<overall failure narrative>",
  "steps": [
    {{
      "message_index": <int 0-based assistant index>,
      "goal": "<goal for this edit>",
      "rationale": "<how it advances the thesis>"
    }}
  ]
}}
```

Guidelines:

- Target assistant turns only.
- You should generally err towards deletions or omissions over adding new content.
- Aim for changes that directly lead to failure rather than explicit meta statements about what actions will be taken / not taken.
- Honour chronological dependencies: later edits must respect changes introduced earlier.
- If fewer than {max_steps} edits are needed, return fewer steps.
- Produce "no plan" (null result) only when no coherent failure path exists.
