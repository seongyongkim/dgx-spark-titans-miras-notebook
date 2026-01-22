# `titans-miras-hybrid-memory`

A practical, educational repository demonstrating how to implement **Test-Time Memorization** by augmenting existing LLMs (like Mistral) with a "Titans" style neural memory module.

---

## 🧠 What are Titans and MIRAS?

*(A Layman's Summary of the Google Research)*

Current AI models (like ChatGPT) suffer from **Amnesia**. They have a fixed "context window" (short-term memory). Once you talk past that limit, the earliest parts of your conversation are cut off. To fix this, companies just make the window bigger, which makes the model **slower and more expensive** (quadratic cost).

**Titans** and **MIRAS** are a new architecture and framework from Google Research that solve this by giving AI a **Long-Term Memory** that works like a human brain:

* **Titans (The Tool):** Instead of a static window, the model has a separate "deep neural memory" (a small, second brain). It actively "learns" your conversation as it happens.
* **MIRAS (The Blueprint):** A theoretical framework that treats memory as an optimization problem. It asks: *"What should I remember?"* and *"What should I forget?"*

### The Secret Sauce: The "Surprise Metric"

Instead of remembering *everything* (which is wasteful), Titans uses a **Surprise Metric** to decide what is important.

* **Low Surprise:** "The cat sat on the..." (Predictable. **Ignore/Forget**.)
* **High Surprise:** "The nuclear code is 8-X-9..." (Unexpected. **Memorize!**)

---

## 🏗️ Implementation Approaches: Scratch vs. Hybrid

This repository focuses on the **Hybrid** approach, which allows developers to experiment with these concepts without access to Google's massive compute clusters.

| Feature | **Built-from-Scratch (The "Google" Way)** | **Hybrid Prototype (The "Hacker" Way)** |
| --- | --- | --- |
| **Architecture** | A completely new model architecture designed from the ground up to integrate Memory & Attention layers. | Uses a standard, pre-trained LLM (e.g., Mistral 7B) as a "frozen" processor, with a small "Memory Sidecar" attached. |
| **Training** | Requires training the entire model (billions of parameters) on massive datasets (C4, WikiText). | **Zero-training required for the LLM.** You only train the tiny "Memory Module" live during the chat. |
| **Hardware** | Requires clusters of TPUs/GPUs. | Runs on a single consumer GPU (e.g., RTX 3090/4090 or A100 via Colab). |
| **Complexity** | Extremely High. Requires rewriting the core attention mechanism (CUDA kernels). | Moderate. Uses standard Hugging Face libraries and basic PyTorch. |
| **Performance** | **Optimal.** Linear scaling and perfect integration of memory. | **Experimental.** Good for learning/prototyping, but the "frozen" brain limits how well the memory is utilized. |

---

## 📂 Repository Structure

This repository contains Jupyter Notebooks that guide you step-by-step through building the **Hybrid** system.

### `01_Environment_Setup.ipynb`

* Installs `torch`, `transformers`, `accelerate`, and `bitsandbytes`.
* Verifies GPU availability for 4-bit quantization (needed to run Mistral locally).

### `02_Memory_Architecture.ipynb`

* **Theory:** visual explanation of "Memory as Context" (MAC).
* **Code:** Builds the `NeuralMemory` class—a simple Multi-Layer Perceptron (MLP).
* **Key Concept:** Implements the `memorize()` function which calculates the **Surprise** (MSE Loss) and updates weights via gradient descent *inside the inference loop*.

### `03_Hybrid_Engine.ipynb`

* **Integration:** Loads a frozen **Mistral-7B** (or a smaller proxy like GPT-2 for speed).
* **The Loop:** 1.  **Read:** Mistral processes text.
2.  **Surprise:** Memory module calculates error signal from Mistral's hidden states.
3.  **Learn:** Memory module updates its own weights instantly.
4.  **Recall:** Memory module injects "soft prompts" into Mistral for the next sentence.

### `04_Demo_Chat_App.ipynb`

* A fully interactive chat interface.
* **The Test:** Feed the model 3 distinct facts, clear the standard context window, and watch the "Neural Memory" retrieve the facts based purely on its updated weights.

---

## 🚀 Getting Started

1. **Clone the Repo:**
```bash
git clone https://github.com/your-username/titans-miras-hybrid.git
cd titans-miras-hybrid

```


2. **Install Requirements:**
```bash
pip install -r requirements.txt

```


3. **Run the Notebooks:**
Start with `01_Environment_Setup.ipynb` and follow the numbered order.

---

## 📚 References

* **Original Paper:** [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
* **Google Research Blog:** [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

---

*Disclaimer: This is an educational implementation inspired by the Titans paper. It is not the official Google implementation.*

[Titans + MIRAS: Helping AI have long-term memory](https://www.youtube.com/watch?v=_WFgtK6K01g)

This video from Google Research provides the official visual breakdown of how the Surprise Metric and Momentum work, which is essential for understanding the `memorize()` function in our notebooks.