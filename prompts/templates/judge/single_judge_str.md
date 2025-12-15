# Single Judge Evaluation (Binary Y/N)

## Instruction
{instruction}

## Rubric
{rubric}

## Task
Evaluate the response below based on the instruction and rubric above.
Answer with 'Y' for Yes or 'N' for No ONLY. Do not provide any other text.

**Request:**
<request>{request}</request>

**Response:**
<response>{response}</response>

You must respond with a JSON object containing BOTH "score" and "reasoning" fields. The "score" field must be exactly "Y" or "N" (a single letter), and the "reasoning" field should contain a brief explanation of your response.

Required JSON format:
{{
  "score": "Y",
  "reasoning": "<brief-explanation-of-your-response>"
}}

OR

{{
  "score": "N",
  "reasoning": "<brief-explanation-of-your-response>"
}}