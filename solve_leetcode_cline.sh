#!/usr/bin/env bash

#### get the input file for the LeetCode problem promptand interface
INPUT_FILE="${1:-input_file}"
[[ -f "$INPUT_FILE" ]] || { echo "❌ file not found: $INPUT_FILE"; exit 1; }

PROMPT_CONTENT="$(cat "$INPUT_FILE")"

#### system policy for cline to solve tasks
POLICY="$(cat <<'EOF'
System instruction:
- DO NOT invoke any tools.
- DO NOT propose to run tools or commands.
- Produce plain text only.
- If tests/commands are needed, OUTPUT THEM AS TEXT ONLY; DO NOT EXECUTE.
- After unit tests pass once conceptually, DO NOT re-run tests; finalize the task.
- The solution script, test script and final report are for the output(**must be output products in the working directory**).
- Also **must** generate edge case test cases for the edge solution script.
- Never ask the user questions, never elicit input, never wait for confirmation.
- If information is missing, make reasonable assumptions and proceed; state them briefly.
- Assume YES to any confirmation.
- Produce the final deliverables in a SINGLE response (single-shot), plain text only.
EOF
)"

#### full user prompt message for cline
FULL_MSG="${POLICY}

${PROMPT_CONTENT}"

#### clean cline instance and start a new one each time
echo "==> Cleaning up"
cline instance kill -a || true

echo "==> Starting instance"
#### cline execution
cline -o "$FULL_MSG"
