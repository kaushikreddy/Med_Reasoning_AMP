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
- The GRPO fine-tuning process is compute-intensive (~5 hours).
- **To test the model quickly**, skip fine-tuning and use the **pre-trained checkpoint**.
- Pre-trained checkpoints are included for demo purposes.

---

## Next Steps
- Experiment by modifying reward functions to fit your specific use case.
- Try swapping datasets and models to test generalization.

---

 **Tip:** This AMP is designed for flexibility—tinker safely without triggering expensive compute unless you explicitly choose to fine-tune!
