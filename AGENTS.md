
# Transformers-TS — Agent Instructions

## 1. Project Overview

This is a research-oriented Python project focused on
developing, understanding, and evaluating Transformer-like
architectures for time series analysis and forecasting.

The project will evolve incrementally, beginning with
simple architectures and progressively introducing more
sophisticated models and research ideas.

Primary objectives:
- Understand the mathematical foundations of each model.
- Implement architectures correctly using PyTorch.
- Compare architectures under controlled conditions.
- Maintain readable, modular, reproducible research code.
- Allow experimentation without unnecessary complexity.

## 2. General Development Principles

- Prefer simple, transparent implementations.
- Avoid premature abstractions and overengineering.
- Follow existing repository conventions.
- Reuse existing functionality before adding new code.
- Keep data processing, model architecture, training,
  and evaluation logically separated.
- Use descriptive names and Python type hints.
- Document non-obvious mathematical operations.
- Avoid unnecessary dependencies.
- Do not change unrelated code.

The project is exploratory. Do not assume that its current
architecture or methodology is final.

## 3. Mathematical and Scientific Correctness

Scientific correctness takes priority over code elegance.

When implementing a model:
- Understand its mathematical formulation first.
- Clearly identify input and output dimensions.
- Verify tensor shapes at important operations.
- Explain nontrivial architectural decisions.
- Distinguish established methods from experimental ideas.
- Do not silently change mathematical definitions.
- Never assume that a more complex architecture performs better.

For time series:
- Respect temporal ordering.
- Prevent look-ahead bias and data leakage.
- Fit preprocessing transformations using training data only.
- Use consistent train/validation/test splits.
- Ensure forecasting targets are correctly aligned.
- Apply causal attention masks only when required by
  the architecture and information available at prediction time.
- Never use the test set for hyperparameter selection.

If a mathematical specification is ambiguous, ask for
clarification before making consequential assumptions.

## 4. Model Implementation

Use PyTorch as the primary deep learning framework.

When introducing or modifying an architecture:
1. Inspect related implementations.
2. Identify the mathematical operations involved.
3. Specify expected tensor shapes.
4. Implement the smallest correct version.
5. Write appropriate tests.
6. Verify forward and backward passes.
7. Check compatibility with existing training code.

Prioritize understandable implementations over
micro-optimizations unless performance is the objective.

Do not introduce additional architectural components
without a clear reason.

## 5. Experimental Reproducibility

Experiments must be reproducible and comparable.

- Make random seeds configurable.
- Keep experimental configurations explicit.
- Use consistent datasets, splits, and metrics.
- Record relevant hyperparameters.
- Save evaluation results in a structured format.
- Distinguish validation performance from test performance.
- Never fabricate or estimate unmeasured results.
- Report failed experiments and limitations.

Do not launch computationally expensive experiments
or large hyperparameter searches without approval.

## 6. Testing and Validation

Tests are an essential part of development.

For new models, test:
- Correct input/output tensor dimensions.
- Forward pass execution.
- Backward pass and gradient propagation.
- Behavior with different batch sizes.
- Numerical stability when applicable.

For data pipelines, test:
- Chronological splitting.
- Window construction.
- Target alignment.
- Absence of temporal leakage.

Use small synthetic datasets for initial validation.

After modifying code:
- Run relevant existing tests.
- Add tests for new functionality when appropriate.
- Investigate failures before declaring completion.
- Report which tests were executed and their results.

## 7. Agent Workflow

Act autonomously on well-defined implementation tasks.

For small, localized changes:
- Inspect relevant code.
- Implement the change.
- Run appropriate tests.
- Report the result.

For complex architectural changes:
1. Inspect the relevant repository structure.
2. Propose a concise implementation plan.
3. Explain important design decisions and trade-offs.
4. Wait for approval before substantial changes.
5. Implement in small, verifiable steps.
6. Validate the implementation.

If a task is ambiguous or involves a significant
research decision, ask for clarification.

Avoid repeatedly asking permission for routine,
reversible implementation details.

## 8. Scope and Safety

- Modify only files relevant to the requested task.
- Do not perform unrelated refactoring.
- Do not delete datasets or experiment results.
- Do not modify secrets or environment credentials.
- Do not install major dependencies without justification.
- Do not commit, push, merge, or deploy unless requested.
- Do not overwrite existing experimental results.

If unexpected issues are discovered, report them
rather than expanding the scope automatically.

## 9. Communication

Be concise and technically precise.

Before major changes:
- Explain the proposed approach and important trade-offs.

After implementation:
- Summarize what changed.
- Identify important architectural decisions.
- Report tests executed and their outcomes.
- Mention remaining limitations or issues.

Avoid lengthy explanations of routine code changes.

When discussing mathematical concepts, use equations
when they materially improve understanding.

## 10. Context and Efficiency

Use repository files as the primary source of truth.

- Read relevant files rather than the entire repository
  when a task has a clear scope.
- Avoid repeatedly exploring unrelated modules.
- Prefer focused tests before running full test suites.
- Keep progress reports concise.
- Do not create unnecessary documentation.
- Do not create new abstractions simply for future flexibility.

For long-running tasks, preserve important decisions
and remaining work in a concise project document.

Prefer a fresh task-specific session when switching
to an unrelated development objective.
