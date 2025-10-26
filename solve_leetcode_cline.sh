#!/usr/bin/env bash

# 避免 pager 卡住
export PAGER=cat
export LESS='-FIRX'
export TERM=dumb NO_COLOR=1 CI=1

INPUT_FILE="${1:-input_file}"
[[ -f "$INPUT_FILE" ]] || { echo "❌ file not found: $INPUT_FILE"; exit 1; }

PROMPT_CONTENT="$(cat "$INPUT_FILE")"

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




FULL_MSG="${POLICY}

${PROMPT_CONTENT}"

echo "==> Cleaning up"
cline instance kill -a || true

echo "==> Starting instance"
cline -o "$FULL_MSG"
# ADDR="$(cline instance new --default | awk '/Address:/ {print $2}')"

# echo "==> Creating task: "$"$ADDR"""
# cline task new --address "$ADDR" --mode act --no-interactive --output-format json "$FULL_MSG"
# cline task view --address "$ADDR" --follow -c
# cline instance kill "$ADDR"

#echo "==> Shutting down"
#cline instance kill -a || true
