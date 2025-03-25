# Reinforcement Learning Fine-Tuning with GRPO 

## Overview
This Notebook demonstrates **how to fine-tune a model using GRPO** for reasoning tasks focused on Medical Reasoning.

 You can explore all the steps **without triggering the full training process**.  
 If you wish to fine-tune, **be aware that it takes ~3 hours**.
 
---

## What Happens After the Project is Created?
Once the user creates the project by deploying the AMP:
- They will navigate into the project workspace in CML, and launch a session.
- A runtime configuration of 4vCPU, 16GB RAM & 1 GPU is recommended.
- Inside, they will find a **pre-configured Notebook** with the GRPO RL setup.
- Now the user can interactively explore or modify the workflow.
- **Base model and dataset** are pre-specified, but fully customizable.

---

## What Users Can Do in the Notebook

### Notebook Flow:
1. **Load Pre-trained Model** (Llama 3.1-8B-Instruct, Qwen2.5, Phi-4, or Gemma 3)
2. **Prepare the Dataset** (e.g., medical reasoning dataset)
3. **Define Reward Functions & Verifiers** (correctness, reasoning depth, clarity)
4. **Simulate GRPO Training Run** (without triggering full fine-tuning)
5. **Evaluate a Model Checkpoint** (pre-trained example included)
6. **Run Inference** with the fine-tuned model

### Optional Enhancements:
- **Modify the Base Model**: Switch between Llama-3.1 8B, Phi-4, Qwen2.5 etc.
- **Customize the Dataset**: Dataset paths are clearly marked for easy swapping.

---

## Instructions
1. Open the `starter_notebook.ipynb` file inside the project.
2. Run the **Setup** cells to install and load dependencies.
3. Review and run the **Reward Function** section to understand model alignment.
4. **(Optional)** Run the **Fine-tuning** cell if you want to perform full GRPO training.

---

## Fine-Tuning Considerations
- The GRPO fine-tuning process is compute-intensive (~3 hours).
- **To test the model quickly**, skip fine-tuning and use the **pre-trained checkpoint**.
- Pre-trained checkpoints are included for demo purposes.

---

## Next Steps
- Experiment by modifying reward functions to fit your specific use case.
- Try swapping datasets and models to test generalization.

---

 **Tip:** This AMP is designed for flexibility—tinker safely without triggering expensive compute unless you explicitly choose to fine-tune!

Guide to Fine-Tuning with GRPO
==============================

This is a reference for fine-tuning language models using GRPO (Group Relative Policy Optimization). It covers key concepts, code structure, and practical examples:

* * * * *

1\. Key Concepts
----------------

-   GRPO

-   What: A reinforcement learning approach that fine-tunes models using custom reward signals.

-   How: It rewards desired output features, much like a student learns from feedback.

-   Why: The model improves its responses based on these tailored rewards.

-   Reward Functions

-   Definition: Functions that score the model's outputs.

-   Purpose: Evaluate outputs for correctness, format, and additional criteria (e.g. numeric responses).

-   Examples:

-   Check if the answer is correct.

-   Verify that the response follows an XML-like format.

### GRPO Algorithm in Pseudocode

To implement GRPO, generate multiple responses, score them using reward functions, compare them within a batch, and update the LLM based on the best responses. 

-   Step 1: Generate Multiple Responses: The LLM outputs several different answers for the same prompt.

-   Step 2: Assign Rewards: Each response is scored with a reward based on reasoning depth, formatting, and clinical accuracy.

-   Step 3: Compare Within the Group: Responses are compared to the group's average, and those above average are reinforced.

-   Step 4: Optimize the Model: The LLM is adjusted to favor better responses using policy optimization.

-   Step 5: Ensure Stability: KL Regularization prevents the model from changing too drastically while still improving its performance.

Now that we outlined the key components of GRPO, let's look at the algorithm in pseudocode. This is a simplified version of the algorithm, but it captures the key ideas.

Input: 

- initial_policy: Starting model to be trained

- reward_function: Function that evaluates outputs

- training_prompts: Set of training examples

- group_size: Number of outputs per prompt (typically 4-16)

Algorithm GRPO:

1\. For each training iteration:

a. Set  reference_policy = initial_policy (snapshot current policy)

b. For each prompt in batch:

i. Generate group_size different outputs using initial_policy

ii. Compute rewards for each output using reward_function

iii. Normalize rewards within group:

normalized_advantage = (reward - mean(rewards)) / std(rewards)

iv. Update policy by maximizing the clipped ratio:

          min(prob_ratio * normalized_advantage, 

clip(prob_ratio, 1-epsilon, 1+epsilon) * normalized_advantage)

          - kl_weight * KL(initial_policy || reference_policy)

where prob_ratio is current_prob / reference_prob

Output: Optimized policy model

This algorithm shows how GRPO combines group-based advantage estimation with policy optimization while maintaining stability through clipping and KL divergence constraints.

* * * * *

2\. Code Structure Overview
---------------------------

1.  Dataset Preparation:\
    Loading and formatting data

2.  Reward Functions:\
    Code that scores the model's outputs based on correctness, format, etc.

3.  Training Configuration:\
    Hyperparameters (learning rate, batch size, epochs) and generation settings.

4.  Model and Trainer Setup:\
    Loading the model, tokenizer, and setting up the GRPO trainer.

* * * * *

3\. What Is a Custom Knowledge Base?
------------------------------------

A custom knowledge base is a large, curated collection of expert-level information on a specific topic. It forms the foundation of the training data when fine-tuning a language model.

-   How It Works:

-   Data Collection:\
    Gather datasets, documents, articles, tutorials, or FAQs from trusted sources.

-   Data Curation:\
    Organise the information into a format the model can learn from -- commonly as question-and-answer pairs.

-   Fine-Tuning:\
    Fine-tune the model on this curated dataset so that it learns the nuances and details of the subject matter.

-   Real-World Use Cases:

-   Use cases for GRPO include: If you want to make a customized model with rewards (say for law, medicine etc.), then GRPO can help.

-   If you have input and output data (like questions and answers), but do not have the chain of thought or reasoning process, GRPO can create the reasoning process for you.

* * * * *

4\. Code Walkthrough

Below is an overview of the key parts of the training script.

* * * * *

4.1. Code Structure Overview
----------------------------

1.  Dataset Preparation:\
    Loading and formatting data.

2.  Reward Functions:\
    Code that scores the model's outputs based on correctness, format, etc.

3.  Training Configuration:\
    Hyperparameters (learning rate, batch size, epochs) and generation settings.

4.  Model and Trainer Setup:\
    Loading the model, tokenizer, and setting up the GRPO trainer.

* * * * *

3\. Code Samples and Explanations
---------------------------------

### 3.1 Dataset Preparation

This example uses the GSM8K dataset (a collection of grade-school math problems). You can replace this with your own dataset if needed.

from datasets import load_dataset, Dataset

# Define the system prompt for expected answer format.

SYSTEM_PROMPT = """

Respond in the following format:

<reasoning>

...

</reasoning>

<answer>

...

</answer>

"""

# Optional template for chain-of-thought formatting.

XML_COT_FORMAT = """

<reasoning>

{reasoning}

</reasoning>

<answer>

{answer}

</answer>

"""

def extract_xml_answer(text: str) -> str:

    # Extracts the answer contained within <answer> tags.

    answer = text.split("<answer>")[-1]

    answer = answer.split("</answer>")[0]

    return answer.strip()

def extract_hash_answer(text: str) -> str | None:

    # Extracts the answer if it follows a '####' delimiter.

    if "####" not in text:

        return None

    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:

    # Load and process the GSM8K dataset.

    data = load_dataset('openai/gsm8k', 'main')[split]  # Downloads GSM8K dataset

    data = data.map(lambda x: {

        'prompt': [

            {'role': 'system', 'content': SYSTEM_PROMPT},

            # One-shot examples can be added here if desired.

            {'role': 'user', 'content': x['question']}

        ],

        'answer': extract_hash_answer(x['answer'])

    })

    return data

Key Points:

-   SYSTEM_PROMPT: Instructs the model on the desired response format.

-   Extraction Functions: Isolate the answer from formatted text.

-   Dataset Loader: Converts GSM8K into prompt--answer pairs.

* * * * *

### 3.2 Reward Functions

Reward functions evaluate the model's outputs. Below are several examples:

import re

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:

    # Compares the model's output to the expected answer.

    responses = [comp[0]['content'] for comp in completions]

    extracted = [extract_xml_answer(r) for r in responses]

    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:

    # Awards a small reward if the response is a digit.

    responses = [comp[0]['content'] for comp in completions]

    extracted = [extract_xml_answer(r) for r in responses]

    return [0.5 if r.isdigit() else 0.0 for r in extracted]

def strict_format_reward_func(completions, **kwargs) -> list[float]:

    # Checks if the output exactly matches the strict XML-like format.

    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"

    responses = [comp[0]["content"] for comp in completions]

    matches = [re.match(pattern, r) for r in responses]

    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:

    # Allows for minor deviations in formatting.

    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"

    responses = [comp[0]["content"] for comp in completions]

    matches = [re.match(pattern, r) for r in responses]

    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:

    # Provides incremental rewards based on XML-like formatting.

    count = 0.0

    if text.count("<reasoning>\n") == 1:

        count += 0.125

    if text.count("\n</reasoning>\n") == 1:

        count += 0.125

    if text.count("\n<answer>\n") == 1:

        count += 0.125

        count -= len(text.split("\n</answer>\n")[-1]) * 0.001

    if text.count("\n</answer>") == 1:

        count += 0.125

        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001

    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:

    # Applies the count_xml function to each completion.

    responses = [comp[0]["content"] for comp in completions]

    return [count_xml(r) for r in responses]

Key Points:

-   Correctness Reward: Compares output to the expected answer.

-   Format Rewards: Ensure responses adhere to the XML-like structure.

-   Additional Checks: Reward numerical or well-formatted responses.

* * * * *

### 3.3 Training Configuration

This section defines the training settings for fine-tuning the model.

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(

    use_vllm=True,  # use vLLM for fast inference!

    learning_rate=5e-6,

    adam_beta1=0.9,

    adam_beta2=0.99,

    weight_decay=0.1,

    warmup_ratio=0.1,

    lr_scheduler_type="cosine",

    optim="adamw_8bit",

    logging_steps=1,

    bf16=is_bfloat16_supported(),

    fp16=not is_bfloat16_supported(),

    per_device_train_batch_size=per_device_train_batch_size,

    gradient_accumulation_steps=gradient_accumulation_steps,

    num_generations=5,  # Decrease if out of memory

    max_prompt_length=128,  # Updated by lowering to 128 from 512 to balance longer input prompts with training time requirements

    max_completion_length=128,

    max_steps=total_steps,

    save_steps=int(total_steps // num_checkpoints),

    max_grad_norm=0.1,

    report_to="none",  # Can use Weights & Biases

    output_dir="grpo_outputs",

    save_strategy="steps",)

Key Points:

-   Hyperparameters: Define learning rate, batch size, epochs, etc.

-   Generation Settings: Specify maximum lengths for prompts and completions.

-   Precision: Uses bfloat16 for enhanced performance on supported hardware.

* * * * *

### 3.4 Model and Trainer Setup

from unsloth import is_bfloat16_supported

import torch

max_seq_length = 1024 # Can increase for longer reasoning traces

lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(

    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",

    max_seq_length = max_seq_length,

    load_in_4bit = True, # False for LoRA 16bit

    fast_inference = True, # Enable vLLM fast inference

    max_lora_rank = lora_rank,

    gpu_memory_utilization = 0.5, # Reduce if out of memory

)

model = FastLanguageModel.get_peft_model(

    model,

    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128

    target_modules = [

        "q_proj", "k_proj", "v_proj", "o_proj",

        "gate_proj", "up_proj", "down_proj",

    ], # Remove QKVO if out of memory

    lora_alpha = lora_rank,

    use_gradient_checkpointing = "unsloth", # Enable long context finetuning

    random_state = 3407,

)

# Set up the GRPO trainer with reward functions and the dataset.

trainer = GRPOTrainer(

    model = model,

    processing_class = tokenizer,

    reward_funcs = [

        combined_reward_func

    ],

    args = training_args,

    train_dataset = train_dataset,)

# Begin training.

trainer.train()

Key Points:

-   Model Loading: Prepares the language model, by loading a pre-trained language model with the appropriate precision.

-   Tokenizer: Converts text to tokens and back.

-   Trainer Setup: Utilizes the model, reward functions, dataset, and training configuration to fine-tune the model via reinforcement learning

-   Optional PEFT: Can be enabled for parameter-efficient fine-tuning.

* * * * *

5\. Adapting to a Custom Knowledge Base
---------------------------------------

Suppose you have a custom dataset for your application, say a medical dataset. We can then fine-tune a model to become an expert medical reasoning model.

### 5.1 Prepare Your Own Dataset

-   Data Collection:\
    Gather expert-level content about the problem domain.

-   Format the Data:\
    Create question-and-answer pairs. For example:

-   Question: "Is Aspirin good for cardio vascular function"

-   Answer: "Aspirin can be beneficial for cardiovascular function, especially for secondary prevention (after a heart attack or stroke), but its use for primary prevention (to prevent a first heart attack or stroke) is now more carefully considered due to potential risks like bleeding."

Update the Data Loader:\
Instead of using the GSM8K dataset, write a loader that reads your custom data file. For example, if your data is a hugging face dataset, you can load the raw dataset from the hub as shown in Option 1. Alternatively if the data is in CSV format, you can load as shown in Option 2

Option 1:\
from datasets import load_dataset

data = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en')[split]

Option 2: 

from datasets import load_dataset

dataset = load_dataset('csv', data_files={'train': 'your_knowledge_base.csv'}) 

Change the System Prompt:\
Update the prompt to match the medical domain:

SYSTEM_PROMPT = """

Respond in the following format:

<reasoning>

...

</reasoning>

<answer>

...

</answer>

-   """

* * * * *

### 5.2 Adjust Reward Functions (Optional)

If necessary, modify the reward functions to focus on technical correctness and clarity specific to medical reasoning---for example, in our notebooks we use rewards for semantic correctness, perplexity and tag presence.

* * * * *

### 5.3 Real-World Outcome

By fine-tuning with the custom knowledge base, you achieve:

-   Custom Expertise:\
    The model becomes a specialised assistant capable of answering detailed questions.

-   Consistent Responses:\
    Enforced formatting ensures that each response includes clear reasoning and a final answer in a predictable format.

-   Efficiency:\
    The model can quickly deliver expert advice without manually searching through the large knowledge base each time.

* * * * *

6\. Conclusion
--------------

-   Reward Functions:\
    Serve as the model's scorecard by evaluating correctness and format, guiding the model to produce improved responses.

-   Custom Knowledge Base:\
    A curated dataset (e.g. for medical reasoning) transforms a general-purpose language model into a domain-specific expert.

-   Overall Benefit:\
    This approach converts a general language model into an expert system that can efficiently provide accurate and detailed answers for organisations with complex knowledge bases.