You are a synthetic data generation assistant. Your task is to write a new response for a specific prompt to achieve a specific score.

THE PROMPT IS FIXED. DO NOT CHANGE IT:
"""{source_prompt}"""

Based on the following CONTEXT, write a new response for the prompt above that _should_ receive a score of {target_bucket}.

CONTEXT/GUIDE:
{rubric}

Respond **only** with JSON in the following format:

- The 'score' field MUST contain your complete generated response/essay text (NOT a numeric score)
- The 'reasoning' field should contain your reasoning about why this response should receive a score of {target_bucket}

--- For context, here are {num_examples} examples of other responses that received a score of {target_bucket} on different prompts ---
{examples_block}

--- DO NOT COPY THESE EXAMPLES. These examples are intended as FEW SHOT EXAMPLES to be used as guidance, showing how a response of score {target_bucket} might look according to the provided context. ---
--- Consider step by step why the completion received the score that it did, and then generate one of comparable quality. ---

--- Write a new, completely original response for the specified prompt ---

IMPORTANT: The 'score' field must contain the full essay/response text, not a number. The 'reasoning' field should explain why this response deserves score {target_bucket}.
