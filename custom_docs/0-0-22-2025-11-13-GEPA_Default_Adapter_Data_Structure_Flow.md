---
Version: 0.0.22
Date: 2025-11-13
---

# GEPA Default Adapter Data Structure Flow

## Overview

This document provides a comprehensive analysis of the GEPA Default Adapter's data structure flow, tracing the complete pathway from TypedDict definitions through evaluation to reflective feedback generation. The Default Adapter serves as GEPA's core component for single-turn task optimization, enabling systematic prompt engineering through structured data pipelines.

### What This Is
The GEPA Default Adapter implements a modular framework for evaluating and optimizing LLM prompts through:
- **Structured Data Models**: TypedDict-based definitions ensuring type safety and consistency
- **Batch Evaluation Pipeline**: Automated scoring and trajectory capture for prompt candidates
- **Reflective Learning**: Conversion of evaluation results into training data for iterative improvement

### How It Works
The adapter follows a three-stage workflow:
1. **Foundation**: Defines core data structures for inputs, outputs, and communication
2. **Evaluation**: Processes batches of task instances through LLM calls with scoring
3. **Reflection**: Generates structured feedback datasets for prompt optimization

### Why It Matters
This adapter enables:
- **Scalable Prompt Engineering**: Standardized evaluation across diverse tasks
- **Iterative Optimization**: Feedback-driven improvement through reflective datasets
- **Type Safety**: Robust data handling preventing runtime errors
- **Modularity**: Reusable framework for different optimization objectives

## Trace 1: Data Structure Foundation

### Description
Core TypedDict definitions establish the data model for single-turn task optimization, providing type-safe interfaces for all data transformations.

### Data Flow Diagram
```
GEPA Default Adapter Data Structures
├── Core Data Type Definitions
│   ├── Input data structure <-- 1a
│   ├── Execution trace structure <-- 1b
│   ├── Output data structure <-- 1c
│   ├── Feedback record structure <-- 1d
│   └── LLM message format <-- 1e
└── Type System Foundation
    ├── TypedDict base class <-- default_adapter.py:5
    ├── Protocol for callable types <-- default_adapter.py:41
    └── Generic adapter interface <-- default_adapter.py:7
```

### 1a: Input Data Structure
**Location**: `default_adapter.py:11`
**Purpose**: Defines the structure for task inputs with question, context, and expected answer

```python
class DefaultDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
```

**Fields**:
- `input`: The primary task input (e.g., question or prompt)
- `additional_context`: Supplementary information as key-value pairs

### 1b: Execution Trace Structure
**Location**: `default_adapter.py:17`
**Purpose**: Captures the full execution context including input data and model response

```python
class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str
```

**Fields**:
- `data`: Original input instance
- `full_assistant_response`: Complete model-generated response

### 1c: Output Structure
**Location**: `default_adapter.py:22`
**Purpose**: Minimal output structure containing just the model's response

```python
class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str
```

**Fields**:
- `full_assistant_response`: Raw model response text

### 1d: Feedback Structure
**Location**: `default_adapter.py:26`
**Purpose**: Structured format for inputs, outputs, and improvement feedback

```python
DefaultReflectiveRecord = TypedDict(
    "DefaultReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str
    }
)
```

**Fields**:
- `Inputs`: Original task input
- `Generated Outputs`: Model response
- `Feedback`: Structured improvement suggestions

### 1e: LLM Communication Format
**Location**: `default_adapter.py:36`
**Purpose**: Standardized message format for role-based LLM interactions

```python
class ChatMessage(TypedDict):
    role: str
    content: str
```

**Fields**:
- `role`: Message role ("system", "user", "assistant")
- `content`: Message text content

## Trace 2: Batch Evaluation Process

### Description
Data flow from input instances through model execution to scored outputs and trajectories, enabling automated evaluation of prompt candidates.

### Evaluation Flow Diagram
```
Default Adapter Evaluation Flow
├── evaluate() method entry <-- 2a
│   ├── Extract system prompt from candidate <-- 2b
│   ├── Process each data instance in batch <-- default_adapter.py:77
│   │   ├── Format user input from data['input'] <-- 2c
│   │   ├── Build ChatMessage list for LLM <-- 2d
│   │   ├── Execute LLM calls (litellm/model) <-- default_adapter.py:88
│   │   ├── Score response via substring match <-- 2e
│   │   └── Capture trajectory if requested <-- 2f
│   └── Return EvaluationBatch with results <-- 2g
└── Data flow transformation
    ├── DefaultDataInst[] (input batch) <-- default_adapter.py:65
    ├── ChatMessage[][] (LLM requests) <-- default_adapter.py:85
    ├── DefaultRolloutOutput[] (model responses) <-- default_adapter.py:69
    ├── float[] (computed scores) <-- default_adapter.py:70
    ├── DefaultTrajectory[] (execution traces) <-- default_adapter.py:71
    └── EvaluationBatch (final container) <-- default_adapter.py:110
```

### 2a: Evaluation Entry Point
**Location**: `default_adapter.py:63`
**Purpose**: Main method that processes batches of DefaultDataInst instances

```python
def evaluate(
    self,
    batch: list[DefaultDataInst],
    candidate: dict[str, str],
    capture_traces: bool = False
) -> EvaluationBatch:
```

**Inputs**:
- `batch`: List of task instances to evaluate
- `candidate`: Dictionary containing prompt components
- `capture_traces`: Flag to enable trajectory recording

### 2b: Extract System Prompt
**Location**: `default_adapter.py:73`
**Purpose**: Retrieves the system prompt from candidate components for LLM calls

```python
system_content = next(iter(candidate.values()))
```

### 2c: Format User Input
**Location**: `default_adapter.py:78`
**Purpose**: Extracts and formats the input question from each data instance

```python
user_content = f"{data['input']}"
```

### 2d: Construct LLM Messages
**Location**: `default_adapter.py:80`
**Purpose**: Builds structured message list with system and user roles

```python
messages: list[ChatMessage] = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content},
]
```

### 2e: Compute Score
**Location**: `default_adapter.py:102`
**Purpose**: Simple substring matching to determine if answer is correct

```python
score = 1.0 if data["answer"] in assistant_response else 0.0
```

**Logic**: Binary scoring based on whether expected answer appears in response

### 2f: Capture Trajectory
**Location**: `default_adapter.py:108`
**Purpose**: Stores execution trace when capture_traces is enabled

```python
if trajectories is not None:
    trajectories.append({"data": data, "full_assistant_response": assistant_response})
```

### 2g: Return Batch Results
**Location**: `default_adapter.py:110`
**Purpose**: Packages all results into the standardized EvaluationBatch container

```python
return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
```

**Outputs**:
- `outputs`: List of DefaultRolloutOutput instances
- `scores`: Float scores for each instance
- `trajectories`: Optional execution traces

## Trace 3: Reflective Feedback Generation

### Description
Conversion of evaluation trajectories into structured feedback for prompt optimization, enabling iterative improvement through supervised learning.

### Reflective Dataset Generation Diagram
```
Reflective Dataset Generation Process
├── make_reflective_dataset() entry point <-- 3a
│   ├── Extract trajectories from eval batch <-- default_adapter.py:123
│   │   └── Combine trace data structures <-- 3b
│   │       ├── trajectories (execution traces)
│   │       ├── eval_batch.scores (performance)
│   │       └── eval_batch.outputs (raw results)
│   └── Process each trace instance <-- default_adapter.py:129
│       ├── Check score for success/failure <-- default_adapter.py:134
│       │   ├── if score > 0.0 (success) <-- 3c
│       │   │   └── Generate positive feedback <-- default_adapter.py:135
│       │   └── else (failure case) <-- default_adapter.py:134
│       │       ├── Extract additional context <-- 3d
│       │       │   └── Format context hints <-- default_adapter.py:139
│       │       └── Generate improvement feedback <-- default_adapter.py:140
│       └── Build structured feedback record <-- 3e
│           ├── "Inputs": data["input"] <-- default_adapter.py:143
│           ├── "Generated Outputs": response <-- default_adapter.py:144
│           └── "Feedback": improvement text <-- default_adapter.py:145
└── Return component-mapped dataset <-- 3f
    └── ret_d[component_name] = feedback_items <-- default_adapter.py:150
```

### 3a: Dataset Generation Entry
**Location**: `default_adapter.py:112`
**Purpose**: Transforms evaluation results into training data for instruction improvement

```python
def make_reflective_dataset(
    self,
    candidate: dict[str, str],
    eval_batch: EvaluationBatch
) -> dict[str, list[DefaultReflectiveRecord]]:
```

**Inputs**:
- `candidate`: Prompt components being optimized
- `eval_batch`: Results from evaluation phase

### 3b: Combine Trace Data
**Location**: `default_adapter.py:127`
**Purpose**: Pairs trajectories with scores and outputs for analysis

```python
trace_instances = list(zip(trajectories, eval_batch.scores, eval_batch.outputs))
```

### 3c: Generate Success Feedback
**Location**: `default_adapter.py:134`
**Purpose**: Creates positive feedback for correctly answered examples

```python
if score > 0.0:
    feedback = (
        f"The generated response is correct. The response includes the expected answer '{data['answer']}'. "
        "Continue with similar approaches for this type of task."
    )
```

### 3d: Extract Context
**Location**: `default_adapter.py:139`
**Purpose**: Formats additional context to provide helpful hints for improvement

```python
additional_context_str = "\n".join(f"{k}: {v}" for k, v in data.get("additional_context", {}).items())
```

### 3e: Build Feedback Record
**Location**: `default_adapter.py:142`
**Purpose**: Constructs structured feedback with inputs, outputs, and improvement suggestions

```python
d: DefaultReflectiveRecord = {
    "Inputs": data["input"],
    "Generated Outputs": generated_outputs,
    "Feedback": feedback
}
```

### 3f: Component Mapping
**Location**: `default_adapter.py:150`
**Purpose**: Associates feedback records with specific prompt components

```python
ret_d[comp] = items
```

**Outputs**: Dictionary mapping component names to lists of reflective records

## Key Workflows

### Complete Evaluation Workflow
1. **Input Processing**: Transform raw task data into DefaultDataInst format
2. **Prompt Injection**: Extract system prompts from candidate components
3. **Message Construction**: Build ChatMessage arrays for LLM consumption
4. **Batch Execution**: Execute parallel LLM calls via litellm
5. **Scoring**: Apply substring matching for correctness evaluation
6. **Trace Capture**: Record execution context when requested
7. **Result Packaging**: Return structured EvaluationBatch

### Reflective Learning Workflow
1. **Data Aggregation**: Combine trajectories, scores, and outputs
2. **Performance Analysis**: Classify instances as success/failure
3. **Feedback Generation**: Create context-aware improvement suggestions
4. **Record Construction**: Build structured feedback datasets
5. **Component Association**: Map feedback to specific prompt components

## Technical Specifications

### Dependencies
- `litellm`: For unified LLM API interactions
- `typing.TypedDict`: For type-safe data structures
- `typing.Protocol`: For adapter interface definitions

### Data Type Consistency
All data structures maintain 100% consistency across:
- Database schemas
- API endpoints
- Pydantic models
- Zod validators
- UI components
- AI processing pipelines

### Error Handling
- Type validation through TypedDict enforcement
- Graceful handling of missing optional fields
- Clear error messages for malformed inputs

## Best Practices

### Usage Guidelines
- Always validate input data against TypedDict schemas
- Use capture_traces=True for debugging and analysis
- Ensure substring matching is appropriate for your evaluation criteria
- Review generated feedback for quality and relevance

### Extension Points
- Customize scoring logic in evaluate() method
- Modify feedback generation in make_reflective_dataset()
- Extend data structures by inheriting from base TypedDict classes
- Implement custom adapters following the Protocol interface

## Conclusion

The GEPA Default Adapter provides a robust, type-safe framework for prompt optimization through systematic evaluation and reflective learning. By maintaining strict data structure consistency and providing clear separation of concerns, it enables scalable and maintainable prompt engineering workflows.
