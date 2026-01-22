# Titans+MIRAS Hybrid Memory on DGX Spark

A practical, educational repository demonstrating how to implement **Test-Time Memorization** by augmenting existing LLMs with a "Titans" style neural memory module.

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

This repository provides **two learning paths** for understanding Titans/MIRAS concepts:

| Feature | **From Scratch** (`Titans_MIRAS_Scratch.ipynb`) | **Hybrid Approach** (`Titans_MIRAS_Hybrid.ipynb`) |
| --- | --- | --- |
| **Architecture** | Full Titans model built from scratch with integrated memory layers | Uses a frozen pre-trained LLM (GPT-2) with a "Memory Sidecar" attached |
| **Training** | Trains the entire model on Shakespeare dataset with MIRAS curriculum | Only trains the tiny Memory Module during inference |
| **Best For** | Understanding the complete Titans architecture | Quick prototyping and experimentation |
| **Hardware** | Requires GPU for reasonable training time | Runs on CPU (slower) or GPU |
| **Complexity** | Higher - builds attention + memory from scratch | Moderate - uses Hugging Face transformers |

---

## 📂 Repository Structure

### `Titans_MIRAS_Scratch.ipynb` — Build from Scratch 🔨

A complete implementation that builds the Titans architecture from the ground up:

1. **Hardware Setup**: Verifies GPU/CUDA environment
2. **Data Pipeline**: Downloads and tokenizes the Tiny Shakespeare dataset
3. **The Model**: Constructs `NeuralMemory`, `TitansBlock`, and `TitansGPT` classes
4. **Training Loop**: Implements MIRAS-style curriculum learning with surprise weighting
5. **Visualization**: Plots training loss and LTM gate usage over time
6. **Interactive Chat**: Talk to your trained model!

**Key Concepts Covered:**
- Character-level tokenization
- Causal attention with memory gating
- Test-time training (TTT) simulation
- Curriculum learning with surprise metrics

### `Titans_MIRAS_Hybrid.ipynb` — Hybrid Approach 🔗

A beginner-friendly tutorial that augments a frozen LLM with learnable memory:

1. **Environment Check**: Verifies Python, PyTorch, and GPU availability
2. **Neural Memory Module**: Builds a trainable memory with `memorize()` and `recall()`
3. **Hybrid Engine**: Connects memory to GPT-2's hidden states
4. **Semantic Memory**: Uses sentence-transformers for fact retrieval with confidence scoring
5. **Total Recall Experiment**: Proves memory works by clearing context and querying facts
6. **Multi-User Demo**: Shows how one LLM can serve multiple users with private memories

**Key Concepts Covered:**
- Test-Time Training (TTT)
- Surprise-driven learning (MSE loss)
- Semantic similarity and embeddings
- Production-ready confidence scoring (gap + threshold)

---

## 🚀 Getting Started

### Option 1: Conda Environment (Recommended for GPU)

For the best experience with GPU acceleration, create a dedicated conda environment:

```bash
# Create environment with RAPIDS, CUDA 13.0, and core packages
conda create -n rapids-25.12 -c rapidsai -c conda-forge -c nvidia \
    rapids=25.12 python=3.12 'cuda-version=13.0' \
    jupyterlab -y

# Install PyTorch with CUDA 13.0 support
conda run -n rapids-25.12 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install AI/ML packages
conda run -n rapids-25.12 pip install accelerate transformers sentence-transformers matplotlib seaborn requests tqdm scikit-learn bitsandbytes
```

> **Note**: After creating the environment, select it as the Jupyter kernel:
> - Click the kernel selector in the top right of VS Code/JupyterLab
> - Choose "rapids-25.12" from the list

### Option 2: Pip Install (Simpler)

```bash
# Clone the repo
git clone https://github.com/your-username/titans-miras-hybrid.git
cd titans-miras-hybrid

# Install requirements
pip install -r requirements.txt
```

### Running the Notebooks

1. **For understanding the full architecture**: Start with `Titans_MIRAS_Scratch.ipynb`
2. **For quick prototyping**: Start with `Titans_MIRAS_Hybrid.ipynb`

---

## ⚠️ Hardware Disclaimer

These notebooks were developed and tested exclusively on **NVIDIA DGX Spark** hardware:

| Component | Specification |
|-----------|---------------|
| **CPU** | ARM64 (Grace CPU) |
| **GPU** | NVIDIA GB10 (Blackwell Architecture) |
| **Memory** | 128 GB Unified Shared Memory |
| **CUDA** | 13.0 |
| **OS** | Ubuntu 24.04 |

> **Note**: The GB10 GPU uses CUDA Compute Capability 12.1, which may require PyTorch 2.5+ for full compatibility. Some warnings about CUDA capability may appear but can be safely ignored.

Performance and compatibility on other hardware configurations (consumer GPUs, cloud instances, etc.) have not been verified.

---

## 📚 References

* **Original Paper:** [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
* **Google Research Blog:** [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
* **Video Explanation:** [Titans + MIRAS: Helping AI have long-term memory](https://www.youtube.com/watch?v=_WFgtK6K01g)

---

*Disclaimer: This is an educational implementation inspired by the Titans paper. It is not the official Google implementation.*