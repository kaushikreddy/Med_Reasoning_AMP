Reinforcement Learning Fine-Tuning with GRPO for Medical Reasoning
==================================================================

Overview
--------

This project demonstrates **how to fine-tune a language model using GRPO (Group Relative Policy Optimization)** specifically for medical reasoning tasks. By leveraging custom reward functions and a specialized knowledge base, we transform a general-purpose language model into a domain-specific medical reasoning expert.

* * * * *

Project Workflow
----------------

1.  **Load Pre-trained Model**
2.  **Prepare the Dataset**
3.  **Define Reward Functions & Verifiers**
4.  **Simulate GRPO Training Run**
5.  **Evaluate Model Checkpoint**
6.  **Run Inference with Fine-tuned Model**

* * * * *

Key Features
------------

### Flexible Model Selection

-   Support for multiple base models:
    -   Llama 3.1-8B-Instruct
    -   Qwen2.5
    -   Phi-4
    -   Gemma 3

### Advanced Fine-Tuning Technique

-   **GRPO (Group Relative Policy Optimization)**
    -   Rewards desired output features
    -   Improves model responses through custom feedback
    -   Maintains model stability during optimization

### Customizable Reward Functions

-   Evaluate model outputs for:
    -   Correctness
    -   Response formatting
    -   Clinical accuracy

* * * * *

Getting Started
---------------

### Prerequisites

-   Python 3.8+
-   GPU with sufficient VRAM (recommended: ~48GB)
-   Recommended runtime: 4vCPU, 16GB RAM, 1 GPU

### Installation

1.  Clone the repository
2.  Install dependencies:

```
pip install -r requirements.txt

```

### Project Setup

1.  Open `starter_notebook.ipynb`
2.  Run setup cells to install dependencies
3.  Review reward function configurations
4.  **(Optional)** Trigger full GRPO training

* * * * *

Usage Guide
-----------

### Quick Start

1.  Use pre-trained checkpoint for immediate demos
2.  Customize base model and dataset paths
3.  Adjust reward functions as needed

### Fine-Tuning Workflow

1.  Prepare your custom knowledge base
2.  Define reward functions
3.  Configure training parameters
4.  Run GRPO training

### Example: Custom Knowledge Base Preparation

```
from datasets import load_dataset

# Load medical reasoning dataset
medical_data = load_dataset('your/medical/dataset')

# Prepare system prompt
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

```

* * * * *

Fine-Tuning Considerations
--------------------------

-   Compute-intensive process (approximately 3 hours)
-   Pre-trained checkpoints available for quick testing
-   Fully customizable workflow

* * * * *

Reward Function Examples
------------------------

-   **Correctness Reward**: Compare model output to expected answer
-   **Format Reward**: Ensure XML-like response structure
-   **Numerical Reward**: Validate numerical responses

* * * * *

Advanced Customization
----------------------

-   Modify base models
-   Create custom reward functions
-   Adapt to specific medical domains

* * * * *

Performance Optimization
------------------------

-   Uses LoRA for parameter-efficient fine-tuning
-   Supports mixed-precision training (bfloat16/fp16)
-   Configurable batch sizes and gradient accumulation

* * * * *

Recommended Next Steps
----------------------

-   Experiment with different reward function designs. The notebook contains examples with semantic correctness, perplexity and tag presence.
-   Test model performance across various medical reasoning scenarios
-   Explore domain-specific fine-tuning techniques
