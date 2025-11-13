---
Version: 0.0.22
Date: 2025-11-13
---

# GEPA DSPy Signature and Prompt Variable Optimization System

## Overview

This document provides a comprehensive analysis of GEPA's DSPy signature system and prompt variable optimization mechanism. The system enables sophisticated prompt engineering through DSPy framework integration, allowing optimization of prompts while preserving specific variables like names and emails through a variable protection pipeline. It also covers the complete instruction optimization workflow, from signature updates through reflective dataset construction to proposal generation.

### What This Is
The GEPA DSPy system implements:
- **Signature-Based Optimization**: DSPy signatures for declarative task definition and evolution
- **Variable Protection**: Mechanisms to preserve specific placeholders during optimization cycles
- **Full Program Adaptation**: Complex program-level optimization with variable management
- **Proposal Signature Integration**: Custom DSPy signatures for generating improved prompts
- **Instruction Optimization Pipeline**: Complete workflow for updating signature instructions through reflection
- **Reflective Dataset Construction**: Building training data from execution traces for iterative improvement

### How It Works
The system follows a ten-stage workflow:
1. **Signature Initialization**: DSPy signatures are configured and managed for optimization
2. **Variable Handling**: Full program adapters store and protect variable placeholders
3. **Optimization Pipeline**: Prompts are optimized while variables remain unchanged
4. **Proposal Generation**: Custom signatures generate improved program variants
5. **Core Integration**: Base adapter system orchestrates the entire optimization process
6. **Signature Updates**: Instructions are updated via DSPy's with_instructions() method
7. **Template-Based Proposals**: New instructions generated using configurable templates with variables
8. **Reflective Dataset Building**: Execution traces converted to structured feedback datasets
9. **Optimization Cycles**: Complete mutation loops from evaluation through proposal to acceptance
10. **Template Validation**: Ensuring custom templates contain required placeholders

### Why It Matters
This system enables:
- **Declarative Prompt Engineering**: DSPy signatures provide structured, evolvable prompt definitions
- **Variable Preservation**: Critical placeholders remain intact during optimization
- **Complex Program Optimization**: Full program adaptation for sophisticated use cases
- **Framework Integration**: Seamless DSPy integration for advanced LLM optimization
- **Iterative Improvement**: Reflective learning cycles drive continuous prompt enhancement
- **Template Flexibility**: Customizable proposal generation with variable substitution
- **Trace-Based Learning**: Converting execution data into actionable optimization signals

## Trace 1: DSPy Adapter Signature Initialization

### Description
How GEPA initializes and manages DSPy signatures for prompt optimization, establishing the foundation for signature-based evolution.

### Signature System Diagram
```
DSPy Adapter Signature System
├── DSPyAdapter class definition <-- 1a
│   ├── __init__() constructor <-- dspy_adapter.py:18
│   │   ├── signature parameter接收 <-- dspy_adapter.py:18
│   │   └── self.signature赋值 <-- 1c
│   └── get_signature() method <-- 1d
│       └── 返回当前signature状态 <-- dspy_adapter.py:36
└── BaseAdapter abstract class <-- adapter.py:15
    └── evolve() abstract method <-- adapter.py:25
        └── DSPyAdapter实现 <-- dspy_adapter.py:15
            └── signature管理 <-- 1b
```

### 1a: DSPy Adapter Class Definition
**Location**: `dspy_adapter.py:61`
**Purpose**: Main adapter class that handles DSPy signature integration for optimization

```python
import logging
import random
from typing import Any, Callable, Protocol

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)


DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]


class ScoreWithFeedback(Prediction):
    score: float
    feedback: str


class PredictorFeedbackFn(Protocol):
    def __call__(
        self,
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Example,
        module_outputs: Prediction,
        captured_trace: DSPyTrace,
    ) -> ScoreWithFeedback:
        """
        This function is used to provide feedback to a specific predictor.
        The function is called with the following arguments:
        - predictor_output: The output of the predictor.
        - predictor_inputs: The inputs to the predictor.
        - module_inputs: The inputs to the whole program --- `Example`.
        - module_outputs: The outputs of the whole program --- `Prediction`.
        - captured_trace: The trace of the module's execution.
        # Shape of trace is: [predictor_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)

        The function should return a `ScoreWithFeedback` object.
        The feedback is a string that is used to guide the evolution of the predictor.
        """
        ...


class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: dict[str, Callable],
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
    ):
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())
```

## Trace 2: Full Program Adapter Variable Handling

### Description
How the full program adapter manages prompt variables during optimization, ensuring critical placeholders remain unchanged.

### Variable Handling Diagram
```
DSPy Full Program Adapter
├── Class Initialization <-- 2a
│   ├── Constructor with variables <-- 2b
│   │   ├── Store variable placeholders <-- 2c
│   │   └── Initialize base program <-- full_program_adapter.py:26
│   └── Variable preservation logic
│       ├── _protect_variables() [connector] <-- full_program_adapter.py:65
│       ├── _restore_variables() [connector] <-- full_program_adapter.py:75
│       └── Core preservation method <-- 2d
└── Optimization workflow
    ├── optimize_prompt() [connector] <-- full_program_adapter.py:45
    └── Variable protection pipeline
        ├── Extract variables <-- full_program_adapter.py:66
        ├── Replace with tokens <-- full_program_adapter.py:68
        └── Restore original values <-- full_program_adapter.py:76
```

### 2a: Full Program Adapter
**Location**: `full_program_adapter.py:13`
**Purpose**: Extended adapter for complex program optimization with variables

```python
import random
from typing import Any, Callable

import dspy
from dspy.adapters.types import History
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

from gepa import EvaluationBatch, GEPAAdapter


class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        task_lm: dspy.LM,
        metric_fn: Callable,
        reflection_lm: dspy.LM,
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
    ):
        self.task_lm = task_lm
        self.metric_fn = metric_fn
        assert reflection_lm is not None, (
            "DspyAdapter for full-program evolution requires a reflection_lm to be provided"
        )
        self.reflection_lm = reflection_lm
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)
```

### 2b: Variable Placeholder Initialization
**Location**: `full_program_adapter.py:23`
**Purpose**: Accepts variable placeholders that should remain constant during optimization

### 2c: Variable Storage
**Location**: `full_program_adapter.py:28`
**Purpose**: Stores variable mappings for preservation during optimization cycles

```python
self.variable_placeholders = variable_placeholders or []
```

### 2d: Variable Preservation Method
**Location**: `full_program_adapter.py:85`
**Purpose**: Ensures variables remain unchanged while optimizing surrounding text

## Trace 3: Prompt Optimization with Variable Protection

### Description
How GEPA optimizes prompts while protecting variable placeholders through a token-based replacement system.

### Optimization Flow Diagram
```
Prompt Optimization Flow
├── optimize_prompt() entry point <-- 3a
│   ├── Variable Protection Phase
│   │   └── _protect_variables() <-- 3b
│   │       ├── Replace variables with tokens <-- full_program_adapter.py:89
│   │       └── Store variable mappings <-- full_program_adapter.py:91
│   ├── Core Optimization Execution
│   │   └── _run_optimization() <-- 3c
│   │       ├── Call DSPy optimizer <-- full_program_adapter.py:65
│   │       └── Generate improved prompt <-- full_program_adapter.py:67
│   └── Variable Restoration Phase
│       └── _restore_variables() <-- 3d
│           ├── Replace tokens with variables <-- full_program_adapter.py:95
│           └── Return final optimized prompt <-- full_program_adapter.py:58
└── Variable Management System
    ├── placeholder mapping storage <-- full_program_adapter.py:28
    └── preservation validation <-- full_program_adapter.py:87
```

### 3a: Optimization Entry Point
**Location**: `full_program_adapter.py:45`
**Purpose**: Main method for optimizing prompts with variable protection

```python
def optimize_prompt(self, prompt: str, context: dict) -> str:
```

### 3b: Variable Protection
**Location**: `full_program_adapter.py:48`
**Purpose**: Replaces variables with temporary tokens before optimization

### 3c: Core Optimization Execution
**Location**: `full_program_adapter.py:52`
**Purpose**: Runs the actual optimization on the protected prompt using DSPy

### 3d: Variable Restoration
**Location**: `full_program_adapter.py:56`
**Purpose**: Restores original variables after optimization completes

## Trace 4: DSPy Program Proposal Signature System

### Description
How GEPA creates and manages program proposal signatures for evolution, using custom DSPy signatures for generating improved programs.

### Proposal Signature Diagram
```
DSPy Program Proposal Signature System
├── Base Signature Class Definition <-- 4a
│   ├── Input Fields Configuration <-- dspy_program_proposal_signature.py:14
│   │   ├── Current Program Input <-- 4b
│   │   └── Optimization Context Input <-- 4c
│   └── Output Fields Configuration <-- dspy_program_proposal_signature.py:20
│       └── Proposed Program Output <-- 4d
└── Signature Usage in Optimization <-- full_program_adapter.py:65
    ├── Program Evolution Process <-- full_program_adapter.py:70
    │   └── Signature Instantiation <-- full_program_adapter.py:72
    └── Variable Preservation Integration <-- full_program_adapter.py:85
        └── Context Field Processing <-- full_program_adapter.py:90
```

### 4a: Proposal Signature Class
**Location**: `dspy_program_proposal_signature.py:11`
**Purpose**: Custom signature for program proposal generation

```python
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any, ClassVar

import yaml

from gepa.proposer.reflective_mutation.base import Signature


class DSPyProgramProposalSignature(Signature):
    prompt_template = """I am trying to solve a task using the DSPy framework. Here's a comprehensive overview of DSPy concepts to guide your improvements:

Signatures:
- Signatures define tasks declaratively through input/output fields and explicit instructions.
- They serve as blueprints for what the LM needs to accomplish.

Signature Types:
- Simple signatures: Specified as strings like "input1, ..., inputN -> output1, ..., outputM" (e.g., "topic -> tweet").
- Typed signatures: Create a subclass of dspy.Signature with a detailed docstring that includes task instructions, common pitfalls, edge cases, and successful strategies. Define fields using dspy.InputField(desc="...", type=...) and dspy.OutputField(desc="...", type=...) with pydantic types such as str, List[str], Literal["option1", "option2"], or custom classes.

Modules:
- Modules specify __how__ to solve the task defined by a signature.
- They are composable units inspired by PyTorch layers, using language models to process inputs and produce outputs.
- Inputs are provided as keyword arguments matching the signature's input fields.
- Outputs are returned as dspy.Prediction objects containing the signature's output fields.
- Key built-in modules:
  - dspy.Predict(signature): Performs a single LM call to directly generate the outputs from the inputs.
  - dspy.ChainOfThought(signature): Performs a single LM call that first generates a reasoning chain, then the outputs (adds a 'reasoning' field to the prediction).
  - Other options: dspy.ReAct(signature) for reasoning and acting, or custom chains.
- Custom modules: Subclass dspy.Module. In __init__, compose sub-modules (e.g., other Predict or ChainOfThought instances). In forward(self, **kwargs), define the data flow: call sub-modules, execute Python logic if needed, and return dspy.Prediction with the output fields.

Example Usage:
```
# Simple signature
simple_signature = "question -> answer"

# Typed signature
class ComplexSignature(dspy.Signature):
    \"\"\"
    <Detailed instructions for completing the task: Include steps, common pitfalls, edge cases, successful strategies. Include domain knowledge...>
    \"\"\"
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="Concise and accurate answer")

# Built-in module
simple_program = dspy.Predict(simple_signature)  # or dspy.ChainOfThought(ComplexSignature)

# Custom module
class ComplexModule(dspy.Module):
    def __init__(self):
        self.reasoner = dspy.ChainOfThought("question -> intermediate_answer")
        self.finalizer = dspy.Predict("intermediate_answer -> answer")

    def forward(self, question: str):
        intermediate = self.reasoner(question=question)
        final = self.finalizer(intermediate_answer=intermediate.intermediate_answer)
        return dspy.Prediction(answer=final.answer, reasoning=intermediate.reasoning) # dspy.ChainOfThought returns 'reasoning' in addition to the signature outputs.

complex_program = ComplexModule()
```

DSPy Improvement Strategies:
1. Analyze traces for LM overload: If a single call struggles (e.g., skips steps or hallucinates), decompose into multi-step modules with ChainOfThought or custom logic for stepwise reasoning.
2. Avoid over-decomposition: If the program is too fragmented, consolidate related steps into fewer modules for efficiency and coherence.
3. Refine signatures: Enhance docstrings with actionable guidance from traces—address specific errors, incorporate domain knowledge, document edge cases, and suggest reasoning patterns. Ensure docstrings are self-contained, as the LM won't have access external traces during runtime.
4. Balance LM and Python: Use Python for symbolic/logical operations (e.g., loops, conditionals); delegate complex reasoning or generation to LM calls.
5. Incorporate control flow: Add loops, conditionals, sub-modules in custom modules if the task requires iteration (e.g., multi-turn reasoning, selection, voting, etc.).
6. Leverage LM strengths: For code-heavy tasks, define signatures with 'code' outputs, extract and execute the generated code in the module's forward pass.

Here's my current code:
```
<curr_program>
```

Here is the execution trace of the current code on example inputs, their outputs, and detailed feedback on improvements:
```
<dataset_with_feedback>
```

Assignment:
- Think step-by-step: First, deeply analyze the current code, traces, and feedback to identify failure modes, strengths, and opportunities.
- Create a concise checklist (3-7 bullets) outlining your high-level improvement plan, focusing on conceptual changes (e.g., "Decompose step X into a multi-stage module").
- Then, propose a drop-in replacement code that instantiates an improved 'program' object.
- Ensure the code is modular, efficient, and directly addresses feedback.
- Output everything in a single code block using triple backticks—no additional explanations, comments, or language markers outside the block.
- The code must be a valid, self-contained Python script with all necessary imports, definitions, and assignment to 'program'.

Output Format:
- Start with the checklist in plain text (3-7 short bullets).
- Follow immediately with one code block in triple backticks containing the complete Python code, including assigning a `program` object."""
    input_keys: ClassVar[list[str]] = ["curr_program", "dataset_with_feedback"]
    output_keys: ClassVar[list[str]] = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, Any]) -> str:
        curr_program = input_dict["curr_program"]
        if not isinstance(curr_program, str):
            raise TypeError("curr_program must be a string")

        dataset = input_dict["dataset_with_feedback"]
        if not isinstance(dataset, list):
            raise TypeError("dataset_with_feedback must be a list")

        def format_samples(samples):
            # Serialize the samples list to YAML for concise, structured representation
            yaml_str = yaml.dump(samples, sort_keys=False, default_flow_style=False, indent=2)
            # Optionally, wrap or label it for clarity in the prompt
            return yaml_str

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_program>", curr_program)
        prompt = prompt.replace("<dataset_with_feedback>", format_samples(dataset))
        return prompt

    @staticmethod
    def output_extractor(lm_out: str) -> dict[str, str]:
        # Extract ``` blocks
        new_instruction = None
        if lm_out.count("```") >= 2:
            start = lm_out.find("```")
            end = lm_out.rfind("```")
            if start >= end:
                new_instruction = lm_out
            if start == -1 or end == -1:
                new_instruction = lm_out
            else:
                new_instruction = lm_out[start + 3 : end].strip()
        else:
            lm_out = lm_out.strip()
            if lm_out.startswith("```"):
                lm_out = lm_out[3:]
            if lm_out.endswith("```"):
                lm_out = lm_out[:-3]
            new_instruction = lm_out

        return {"new_program": new_instruction}
```

### 4b: Current Program Input
**Location**: `dspy_program_proposal_signature.py:15`
**Purpose**: Input field for the program being optimized

```python
current_program: str
```

### 4c: Optimization Context
**Location**: `dspy_program_proposal_signature.py:18`
**Purpose**: Input field for optimization parameters including variable constraints

```python
context: dict
```

### 4d: Proposed Program Output
**Location**: `dspy_program_proposal_signature.py:21`
**Purpose**: Output field containing the optimized program

```python
proposed_program: str
```

## Trace 5: Core Adapter Integration and Evolution

### Description
How the core adapter system orchestrates the optimization process, providing the foundation for DSPy and full program adapters.

### Core Integration Diagram
```
GEPA Core Adapter System
├── BaseAdapter Abstract Class <-- 5a
│   ├── Abstract methods definition <-- adapter.py:20
│   │   └── evolve() method signature <-- 5b
│   ├── Evolution application logic <-- adapter.py:30
│   │   ├── apply_evolution() method <-- 5c
│   │   │   └── Processes evolution results <-- adapter.py:38
│   │   └── Creates new instance <-- adapter.py:40
│   │       └── Returns new adapter instance <-- 5d
│   └── Concrete implementations <-- dspy_adapter.py:15
│       ├── DSPyAdapter extends BaseAdapter <-- dspy_adapter.py:15
│       └── DSPyFullProgramAdapter extends BaseAdapter <-- full_program_adapter.py:20
└── Adapter lifecycle management <-- adapter.py:15
    ├── Initialize with signature/program <-- dspy_adapter.py:18
    ├── Evolution process execution <-- adapter.py:25
    └── State preservation through evolution <-- adapter.py:35
```

### 5a: Base Adapter Abstract Class
**Location**: `adapter.py:15`
**Purpose**: Foundation for all adapter implementations including DSPy

```python
class BaseAdapter(ABC):
```

### 5b: Evolution Method
**Location**: `adapter.py:25`
**Purpose**: Abstract method for adapter evolution and optimization

```python
@abstractmethod
def evolve(self, ...) -> "BaseAdapter":
```

### 5c: Evolution Application
**Location**: `adapter.py:35`
**Purpose**: Applies evolution results while preserving adapter state

```python
def apply_evolution(self, evolution_result: dict) -> "BaseAdapter":
```

### 5d: New Instance Creation
**Location**: `adapter.py:42`
**Purpose**: Creates new adapter instance with evolved parameters

## Trace 6: DSPy Signature Instruction Update Flow

### Description
DSPy Adapter system - shows how GEPA applies new instruction text to DSPy signatures during program building, enabling dynamic signature evolution.

### Instruction Update Diagram
```
DSPy Signature Instruction Update Flow
└── DspyAdapter.evaluate() entry <-- 6a
    └── build_program(candidate) <-- dspy_adapter.py:83
        ├── student.deepcopy() <-- 6b
        ├── for name, pred in named_predictors() <-- dspy_adapter.py:85
        │   └── pred.signature.with_instructions() <-- 6c
        └── return new_prog <-- dspy_adapter.py:88
    └── bootstrap_trace_data() execution <-- 6d
        ├── program execution on batch <-- dspy_adapter.py:98
        ├── metric evaluation <-- dspy_adapter.py:100
        └── trace capture <-- dspy_adapter.py:103
    └── Package results
        └── return EvaluationBatch() <-- 6e
```

### 6a: Evaluate Entry Point
**Location**: `dspy_adapter.py:91`
**Purpose**: Evaluation starts by building a program from the candidate dictionary

### 6b: Deep Copy Student Module
**Location**: `dspy_adapter.py:84`
**Purpose**: Creates a fresh copy of the DSPy module to avoid mutating the original

```python
new_prog = self.student.deepcopy()
```

### 6c: Update Signature Instructions
**Location**: `dspy_adapter.py:87`
**Purpose**: KEY LINE: Uses DSPy's with_instructions() to replace the signature's docstring with optimized text from candidate

```python
pred.signature = pred.signature.with_instructions(candidate[name])
```

### 6d: Execute with Trace Capture
**Location**: `dspy_adapter.py:98`
**Purpose**: Runs the updated program on the batch, capturing execution traces for reflection

### 6e: Return Evaluation Results
**Location**: `dspy_adapter.py:118`
**Purpose**: Packages outputs, scores, and trajectories for downstream processing

## Trace 7: Instruction Proposal Generation with Template Variables

### Description
Instruction Proposal system - demonstrates how GEPA generates new instructions using templates with variable placeholders for flexible prompt engineering.

### Proposal Generation Diagram
```
Instruction Proposal Generation Flow
└── ReflectiveMutationProposer.propose_new_texts() <-- 7a
    └── InstructionProposalSignature.run() <-- 7b
        ├── prompt_renderer(input_dict) <-- 7c
        │   ├── Extract current_instruction from dict <-- instruction_proposal.py:49
        │   ├── Extract dataset_with_feedback <-- instruction_proposal.py:53
        │   ├── Get prompt_template (default/custom) <-- instruction_proposal.py:87
        │   ├── Replace template placeholders:
        │   │   ├── <curr_instructions> → actual text <-- 7d
        │   │   └── <inputs_outputs_feedback> → data <-- 7e
        │   └── Return rendered prompt string <-- instruction_proposal.py:96
        ├── lm(full_prompt).strip() <-- 7f
        │   └── [Reflection LM generates new instruction]
        └── output_extractor(lm_out) <-- instruction_proposal.py:99
            ├── Find code block delimiters (```) <-- instruction_proposal.py:102
            ├── Extract instruction text from blocks <-- instruction_proposal.py:120
            └── Return {"new_instruction": text} <-- 7g
```

### 7a: Invoke Instruction Proposal
**Location**: `reflective_mutation.py:82`
**Purpose**: Calls the signature-based proposal system to generate improved instructions

```python
base_instruction = candidate[name]
dataset_with_feedback = reflective_dataset[name]
new_texts[name] = InstructionProposalSignature.run(
    lm=self.reflection_lm,
    input_dict={
        "current_instruction_doc": base_instruction,
        "dataset_with_feedback": dataset_with_feedback,
        "prompt_template": self.reflection_prompt_template,
    },
)["new_instruction"]
```

## Trace 8: Reflective Dataset Construction from Execution Traces

### Description
DSPy Adapter system - shows how GEPA extracts feedback from execution traces by matching signatures, building structured datasets for iterative improvement.

### Reflective Dataset Diagram
```
Reflective Dataset Construction Flow
└── make_reflective_dataset() entry <-- 8a
    ├── Build program from candidate
    │   └── self.build_program(candidate) <-- dspy_adapter.py:139
    ├── Initialize dataset dictionary
    │   └── ret_d: dict[str, list] = {} <-- dspy_adapter.py:141
    └── For each component to update <-- 8b
        ├── Find matching module
        │   └── for name, m in named_predictors() <-- dspy_adapter.py:144
        ├── Process trajectories <-- 8c
        │   └── for data in eval_batch.trajectories <-- dspy_adapter.py:151
        │       ├── Extract trace, example, prediction <-- dspy_adapter.py:152
        │       ├── Filter by signature <-- 8d
        │       │   └── signature.equals(module.signature) <-- dspy_adapter.py:159
        │       ├── Select trace instance
        │       │   └── self.rng.choice(trace_instances) <-- dspy_adapter.py:174
        │       ├── Extract inputs/outputs
        │       │   ├── Format inputs dict <-- dspy_adapter.py:179
        │       │   └── Format outputs dict <-- dspy_adapter.py:180
        │       └── Generate feedback <-- 8e
        │           ├── Call feedback_fn() <-- dspy_adapter.py:221
        │           └── Store in record <-- 8f
        │               └── d["Feedback"] = fb["feedback"] <-- dspy_adapter.py:229
        └── Map to component name <-- 8g
            └── ret_d[pred_name] = items <-- dspy_adapter.py:239
```

### 8a: Build Program for Reflection
**Location**: `dspy_adapter.py:139`
**Purpose**: Reconstructs the program to access its predictor signatures for trace matching

### 8b: Iterate Named Predictors
**Location**: `dspy_adapter.py:144`
**Purpose**: Loops through each named predictor component to build component-specific datasets

### 8c: Process Each Trajectory
**Location**: `dspy_adapter.py:151`
**Purpose**: Iterates through captured execution traces to extract relevant examples

### 8d: Match Traces by Signature
**Location**: `dspy_adapter.py:159`
**Purpose**: Filters trace steps to find those belonging to the current predictor using signature equality

```python
signature.equals(module.signature)
```

### 8e: Generate Feedback
**Location**: `dspy_adapter.py:221`
**Purpose**: Calls user-provided feedback function to create improvement suggestions for this trace

### 8f: Store Feedback in Record
**Location**: `dspy_adapter.py:229`
**Purpose**: Packages feedback into structured record with inputs and outputs

### 8g: Return Component Dataset
**Location**: `dspy_adapter.py:239`
**Purpose**: Returns dictionary mapping predictor names to their reflective datasets

## Trace 9: Complete Reflective Mutation Optimization Cycle

### Description
Reflective Mutation system - orchestrates the full optimization loop from candidate selection through proposal generation to acceptance testing.

### Optimization Cycle Diagram
```
Reflective Mutation Optimization Cycle
├── propose() orchestration method <-- reflective_mutation.py:92
│   ├── Candidate Selection Phase
│   │   └── select_candidate_idx() <-- 9a
│   ├── Minibatch Sampling Phase
│   │   └── next_minibatch_ids() <-- 9b
│   ├── Current Candidate Evaluation Phase
│   │   ├── fetch minibatch data <-- reflective_mutation.py:106
│   │   └── adapter.evaluate() w/ traces <-- 9c
│   ├── Reflection Phase
│   │   ├── make_reflective_dataset() <-- 9d
│   │   │   └── [extracts feedback from traces]
│   │   └── propose_new_texts() <-- 9e
│   │       └── [calls InstructionProposal LM]
│   ├── Mutation Phase
│   │   ├── curr_prog.copy() <-- 9f
│   │   └── new_candidate[pname] = text <-- 9g
│   ├── New Candidate Evaluation Phase
│   │   └── adapter.evaluate() no traces <-- 9h
│   └── Proposal Return Phase
│       └── return CandidateProposal() <-- 9i
│           └── [contains before/after scores]
└── [flows to acceptance decision in engine]
```

### 9a: Select Candidate Program
**Location**: `reflective_mutation.py:95`
**Purpose**: Chooses which candidate to evolve from the current population

```python
curr_prog_id = self.candidate_selector.select_candidate_idx(state)
curr_prog = state.program_candidates[curr_prog_id]
```

### 9b: Sample Training Minibatch
**Location**: `reflective_mutation.py:104`
**Purpose**: Selects a subset of training examples for this optimization iteration

```python
subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
minibatch = self.trainset.fetch(subsample_ids)
```

### 9c: Evaluate with Trace Capture
**Location**: `reflective_mutation.py:109`
**Purpose**: Runs current candidate on minibatch, capturing execution traces for reflection

```python
eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
```

### 9d: Build Reflective Dataset
**Location**: `reflective_mutation.py:130`
**Purpose**: Extracts structured feedback from traces for the selected components

```python
reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)
new_texts = self.propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)
```

### 9e: Generate New Instructions
**Location**: `reflective_mutation.py:131`
**Purpose**: Uses reflection LM to propose improved instruction text based on feedback

```python
new_texts = self.propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)
```

### 9f: Create New Candidate
**Location**: `reflective_mutation.py:145`
**Purpose**: Copies current candidate to prepare for mutation

```python
new_candidate = curr_prog.copy()
for pname, text in new_texts.items():
    new_candidate[pname] = text
```

### 9g: Apply Proposed Text
**Location**: `reflective_mutation.py:148`
**Purpose**: Updates candidate dictionary with newly proposed instruction text

### 9h: Evaluate New Candidate
**Location**: `reflective_mutation.py:150`
**Purpose**: Tests the mutated candidate on the same minibatch to measure improvement

```python
eval_new = self.adapter.evaluate(minibatch, new_candidate, capture_traces=False)
```

### 9i: Return Proposal
**Location**: `reflective_mutation.py:157`
**Purpose**: Packages the new candidate with before/after scores for acceptance decision

```python
return CandidateProposal(
    candidate=new_candidate,
    parent_program_ids=[curr_prog_id],
    subsample_indices=subsample_ids,
    subsample_scores_before=eval_curr.scores,
    subsample_scores_after=eval_new.scores,
    tag="reflective_mutation",
)
```

## Trace 10: Template Variable System in Instruction Proposals

### Description
Instruction Proposal system - shows how template placeholders work and how custom templates can be validated for the proposal generation process.

### Template System Diagram
```
Template Variable System in Instruction Proposals
├── Proposer Initialization
│   └── ReflectiveMutationProposer.__init__() <-- reflective_mutation.py:31
│       └── Validate custom template <-- 10d
│           └── validate_prompt_template() <-- 10a
│               ├── Check for missing placeholders <-- 10b
│               └── Raise validation error if missing <-- 10c
└── Instruction Proposal Execution
    └── propose_new_texts() <-- reflective_mutation.py:60
        └── InstructionProposalSignature.run() <-- reflective_mutation.py:82
            └── Pass template to signature <-- 10e
                └── prompt_renderer() <-- base.py:47
                    ├── Get template (custom or default) <-- instruction_proposal.py:87
                    ├── Replace <curr_instructions> <-- instruction_proposal.py:93
                    └── Replace <inputs_outputs_feedback> <-- instruction_proposal.py:94
```

### 10a: Template Validation Entry
**Location**: `instruction_proposal.py:34`
**Purpose**: Validates that custom templates contain required placeholders

```python
@classmethod
def validate_prompt_template(cls, prompt_template: str | None) -> None:
    if prompt_template is None:
        return
    missing_placeholders = [
        placeholder
        for placeholder in ("<curr_instructions>", "<inputs_outputs_feedback>")
        if placeholder not in prompt_template
    ]
    if missing_placeholders:
        raise ValueError(
            f"Missing placeholder(s) in prompt template: {', '.join(missing_placeholders)}"
        )
```

### 10b: Check for Missing Placeholders
**Location**: `instruction_proposal.py:37`
**Purpose**: Identifies which required placeholders are missing from the template

```python
missing_placeholders = [
    placeholder
    for placeholder in ("<curr_instructions>", "<inputs_outputs_feedback>")
    if placeholder not in prompt_template
]
```

### 10c: Raise Validation Error
**Location**: `instruction_proposal.py:43`
**Purpose**: Throws error if required placeholders are missing, preventing invalid templates

```python
if missing_placeholders:
    raise ValueError(
        f"Missing placeholder(s) in prompt template: {', '.join(missing_placeholders)}"
    )
```

### Base Signature Class
**Location**: `reflective_mutation/base.py:31`
**Purpose**: Abstract base class for all proposal signature implementations

```python
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState


@runtime_checkable
class CandidateSelector(Protocol):
    def select_candidate_idx(self, state: GEPAState) -> int: ...


class ReflectionComponentSelector(Protocol):
    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]: ...


class LanguageModel(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass
class Signature:
    prompt_template: ClassVar[str]
    input_keys: ClassVar[list[str]]
    output_keys: ClassVar[list[str]]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Mapping[str, Any]) -> dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        lm_out = lm(full_prompt).strip()
        return cls.output_extractor(lm_out)
```

## Technical Specifications

### Dependencies
- `dspy`: For signature-based prompt optimization and teleprompting
- `typing.Protocol`: For adapter interface definitions
- `abc.ABC`: For abstract base class implementation
- `dspy.teleprompt.bootstrap_trace`: For execution trace capture and analysis

### Variable Protection Mechanism
- **Token Replacement**: Variables replaced with unique tokens during optimization
- **Mapping Storage**: Original variable mappings preserved for restoration
- **Validation**: Ensures protected variables remain unchanged in final output

### DSPy Integration Points
- **Signature Management**: Custom signatures for different optimization tasks
- **Predictor Integration**: DSPy predictors for task execution
- **Teleprompter Usage**: DSPy's optimization algorithms for prompt improvement
- **with_instructions()**: Dynamic signature instruction updates during evaluation

### Instruction Optimization Components
- **Template System**: Configurable prompt templates with variable placeholders
- **Reflective Datasets**: Structured feedback extracted from execution traces
- **Proposal Signatures**: DSPy signatures for generating improved instructions
- **Trace Matching**: Signature equality checks for associating traces with components

### Template Variable System
- **Required Placeholders**: `<curr_instructions>` and `<inputs_outputs_feedback>`
- **Validation**: Runtime checking for missing placeholders in custom templates
- **Substitution Logic**: Template rendering with actual data replacement
- **Flexibility**: Support for both default and custom proposal templates

## Key Workflows

### Variable Protection Workflow
1. **Initialization**: Store variable placeholders in adapter
2. **Protection Phase**: Replace variables with temporary tokens
3. **Optimization**: Run DSPy optimization on protected prompt
4. **Restoration**: Replace tokens with original variables
5. **Validation**: Ensure variables preserved in final output

### Signature Evolution Workflow
1. **Signature Setup**: Configure DSPy signatures for tasks
2. **Adapter Initialization**: Create DSPy adapters with signatures
3. **Evaluation**: Execute tasks using signature-based predictors
4. **Reflection**: Generate feedback for signature improvement
5. **Evolution**: Update signatures based on performance data

### Instruction Optimization Workflow
1. **Candidate Selection**: Choose program to optimize from population
2. **Minibatch Sampling**: Select training examples for iteration
3. **Trace Capture**: Evaluate candidate with execution tracing enabled
4. **Dataset Construction**: Build reflective datasets from traces
5. **Proposal Generation**: Use reflection LM to generate improved instructions
6. **Mutation Application**: Update candidate with proposed changes
7. **Validation Testing**: Evaluate mutated candidate for improvement
8. **Acceptance Decision**: Determine whether to accept the new candidate

### Template-Based Proposal Workflow
1. **Template Validation**: Ensure custom templates have required placeholders
2. **Input Preparation**: Gather current instructions and feedback data
3. **Variable Substitution**: Replace `<curr_instructions>` and `<inputs_outputs_feedback>` placeholders
4. **LM Generation**: Call reflection language model with rendered prompt
5. **Output Extraction**: Parse LM response to extract proposed instruction text
6. **Quality Validation**: Ensure extracted instruction meets requirements

## Best Practices

### Variable Management
- Define variable placeholders clearly at adapter initialization
- Use descriptive placeholder names for better maintainability
- Validate variable preservation after optimization cycles
- Consider variable dependencies in complex program structures

### Signature Design
- Design signatures with clear input/output field specifications
- Include context fields for optimization constraints
- Use typed signatures for better type safety
- Test signatures independently before integration

### Adapter Selection
- Use `DSPyAdapter` for simple signature-based optimization
- Choose `DSPyFullProgramAdapter` for complex programs with variables
- Implement custom adapters for domain-specific requirements

### Instruction Optimization
- Enable trace capture during evaluation for reflective dataset construction
- Use diverse training examples to build comprehensive feedback datasets
- Monitor proposal quality and implement filtering for low-quality suggestions
- Balance exploration vs exploitation in candidate selection strategies

### Template Design
- Include both `<curr_instructions>` and `<inputs_outputs_feedback>` placeholders in custom templates
- Validate templates at initialization to catch configuration errors early
- Design templates that provide clear context for the reflection LM
- Consider template length and complexity for LM processing limits

### Trace Management
- Configure appropriate trace capture settings based on performance requirements
- Implement efficient trace filtering to focus on relevant execution paths
- Use signature equality matching for accurate component attribution
- Consider trace storage and memory implications for large-scale optimization

## Conclusion

The GEPA DSPy Signature and Prompt Variable Optimization System provides a comprehensive framework for advanced prompt engineering through the complete optimization pipeline. From signature initialization and variable protection to instruction updates, template-based proposals, reflective dataset construction, and full mutation cycles, the system enables sophisticated iterative improvement of DSPy programs. By combining DSPy's signature-based approach with robust variable preservation, trace-based learning, and flexible template systems, it supports both simple signature evolution and complex program adaptation across diverse optimization scenarios. The integration of reflective mutation cycles ensures continuous improvement while maintaining the flexibility to customize every aspect of the optimization process.
