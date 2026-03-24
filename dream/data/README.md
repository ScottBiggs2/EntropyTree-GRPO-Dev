# Dream Code GRPO Sample Data

These files provide a tiny task schema for the Dream code-GRPO path.

Each JSONL row is a single task with:

- `task_id`
- `source`
- `split`
- `instruction`
- `starter_code`
- `entry_point`
- `tests`

`canonical_prompt` is optional; if omitted it is built from the instruction and
starter code by `dream.src.task_registry`.

The sample files are intentionally tiny and exist for:

- local unit tests,
- runner smoke tests,
- schema examples for future task exporters.
