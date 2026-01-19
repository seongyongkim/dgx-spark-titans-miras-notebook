**Role:** Expert AI Research Engineer & Educator
**Task:** Generate a comprehensive Jupyter Notebook (`.ipynb`) demonstrating how to implement a "Hybrid" Titans Memory architecture using PyTorch and Hugging Face Transformers.
**Objective:** Show how to augment a frozen LLM (e.g., Mistral-7B or a smaller proxy like GPT-2/TinyLlama for execution speed) with a trainable "Neural Memory" sidecar that learns in real-time via Test-Time Training (TTT).
**Notebook Structure & Requirements:**
1. **Environment Setup:** Install necessary libraries (`torch`, `transformers`, `accelerate`, `bitsandbytes` for 4-bit loading).
2. **Architecture Design (The "Why"):** Briefly explain the "Memory as Context" concept using Markdown headers.
3. **The Memory Module (`NeuralMemory` Class):**     * Code a Multi-Layer Perceptron (MLP) that compresses hidden states.
    * Implement a `memorize()` function that calculates "Surprise" (MSE Loss between input/output) and performs a **single gradient descent step** to update weights immediately.
4. **The Hybrid Engine:**
    * Load the Main LLM (frozen).
    * Create an inference loop that:
        * Reads input.
        * Injects the current Memory state (soft prompts) into the LLM's embeddings.
        * Extracts the LLM's hidden state.
        * Updates the Memory module based on the "Surprise" of that hidden state.

5. **Demo Application (Chat App):**
    * Build a simple interactive text loop or `ipywidgets` interface.
    * **Scenario:** Simulate a "Session" where the user feeds 3 distinct facts, clears the standard context window, and asks a question that requires the Neural Memory to answer.

**Tone:** Professional, clear, and highly educational with comments explaining the math behind the gradient updates.