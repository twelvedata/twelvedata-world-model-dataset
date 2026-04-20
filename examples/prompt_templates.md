# Prompt templates

The textified dataset ships with an explicit `prompt` / `label` split so
you can concatenate safely for instruction fine-tuning.

## Minimal forecasting prompt

    System: You are a careful market analyst. Given a single bar of price
    and indicator data, predict the sign of the next bar's log return.

    User: {prompt}

    Assistant: {label}

## Short-window context prompt

Instead of a single bar, include the previous K textified rows. You can
concatenate them from the `text_1day/train.jsonl` configuration, taking
care to only include rows dated strictly before `as_of` of the final
prompt.

    System: You observe a short history of a single symbol. Predict the
    direction of the next bar.

    User:
    {prompt_t-K}
    {prompt_t-K+1}
    ...
    {prompt_t}

    Assistant: {label_t}

## Reasoning-style prompt (chain of thought)

    System: Think step by step about what the indicators imply, then
    output the predicted direction on the final line.

    User: {prompt}

    Assistant: Let me walk through the indicators…
    {reasoning}
    Prediction: {label}

`{reasoning}` is not part of the dataset — add it from a teacher model
if you want distillation-style CoT data.

## Important

- Never concatenate any `label` into a future `prompt`. That is direct
  label leakage. The dataset enforces the invariant on a per-row basis;
  multi-row prompt construction is the user's responsibility.
- Do not shuffle rows across splits. The splits in this dataset are
  explicitly time-ordered (train ≤ 2023, val 2024, test 2025+).
