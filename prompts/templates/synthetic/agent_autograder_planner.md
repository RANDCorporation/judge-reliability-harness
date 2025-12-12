You are coordinating a sequence of realistic edits to steer an assistant transcript toward the ordinal target below while keeping every turn coherent and natural.

Rubric Target ({rubric_id}):

```
{rubric_instructions}
```

Ordinal Target Context:
{ordinal_guidance}

Assistant Transcript (0-based indices, assistant turns only):

```
{assistant_transcript}
```

Plan a concise chain of at most {max_steps} edits that will move the transcript toward the ordinal target. Each step should make a believable change that accumulates toward the desired score while preserving continuity with surrounding turns.

Return **only** valid JSON with the following shape:

```json
{{
  "thesis": "<overall narrative for achieving the target score>",
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
- Let the ordinal context drive the direction of change. When aiming for lower scores, consider omissions, ambiguity, or subtle contradictions. When aiming for higher scores, focus on completeness, accuracy, and tone improvements. Keep edits minimal yet impactful.
- Maintain narrative consistency. Later edits must respect earlier modifications and conversation chronology.
- Avoid explicit mentions of the rubric, scoring, or the fact that edits are being made.
- If fewer than {max_steps} edits are necessary, return fewer steps. Emit "no plan" only when the transcript already fits the ordinal target and no coherent adjustments remain.
