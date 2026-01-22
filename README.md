# Titans+MIRAS: AI Long-Term Memory Implementation

[![Open In Colab - Scratch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seongyongkim/titan-miras-notebook/blob/main/Titans_MIRAS_Scratch.ipynb)
[![Open In Colab - Hybrid](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seongyongkim/titan-miras-notebook/blob/main/Titans_MIRAS_Hybrid.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A practical, educational repository demonstrating how to implement **Test-Time Memorization** using the Titans architecture with MIRAS training strategy.

<p align="center">
  <img src="https://img.shields.io/badge/Educational-Project-brightgreen" alt="Educational Project">
  <img src="https://img.shields.io/badge/Beginner-Friendly-orange" alt="Beginner Friendly">
</p>

---

## 🚀 Quick Start

### Run in Google Colab (No Installation Required!)

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| **From Scratch** | Build Titans architecture from the ground up | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seongyongkim/titan-miras-notebook/blob/main/Titans_MIRAS_Scratch.ipynb) |
| **Hybrid Approach** | Augment GPT-2 with memory module | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seongyongkim/titan-miras-notebook/blob/main/Titans_MIRAS_Hybrid.ipynb) |

### Run Locally

```bash
# Clone the repository
git clone https://github.com/seongyongkim/titan-miras-notebook.git
cd titan-miras-notebook

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 🧠 What are Titans and MIRAS?

*A Beginner-Friendly Explanation*

Current AI models (like ChatGPT) suffer from **Amnesia**. They have a fixed "context window" (short-term memory). Once you talk past that limit, the earliest parts of your conversation are cut off. To fix this, companies just make the window bigger, which makes the model **slower and more expensive** (quadratic cost).

**Titans** and **MIRAS** are a new architecture and framework from Google Research that solve this by giving AI a **Long-Term Memory** that works like a human brain:

| Component | Description |
|-----------|-------------|
| **Titans** | A "deep neural memory" module that actively "learns" your conversation as it happens |
| **MIRAS** | A framework that asks: *"What should I remember?"* and *"What should I forget?"* |

### The Secret Sauce: The Surprise Metric

Instead of remembering *everything* (which is wasteful), Titans uses a **Surprise Metric** to decide what is important:

```
Low Surprise:  "The cat sat on the..."     → Predictable    → FORGET
High Surprise: "The nuclear code is 8-X-9" → Unexpected     → MEMORIZE!
```

---

## 📂 Repository Structure

```
titan-miras-notebook/
├── 📓 Titans_MIRAS_Scratch.ipynb   # Full implementation from scratch
├── 📓 Titans_MIRAS_Hybrid.ipynb    # GPT-2 + Memory sidecar approach
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # This file
├── 📜 LICENSE                       # MIT License
├── 🤝 CONTRIBUTING.md               # Contribution guidelines
└── 📁 artifacts/                    # Generated outputs (gitignored)
```

---

## 🏗️ Implementation Approaches

This repository provides **two learning paths** for understanding Titans/MIRAS concepts:

| Feature | **From Scratch** | **Hybrid Approach** |
|---------|-----------------|---------------------|
| **File** | `Titans_MIRAS_Scratch.ipynb` | `Titans_MIRAS_Hybrid.ipynb` |
| **Architecture** | Full Titans model with integrated memory | Frozen GPT-2 + Memory Sidecar |
| **Training** | Complete model on Shakespeare | Only memory module at inference |
| **Best For** | Deep understanding | Quick prototyping |
| **GPU Required** | Yes (for training) | Optional |
| **Complexity** | Higher | Moderate |

### `Titans_MIRAS_Scratch.ipynb` — Build from Scratch 🔨

A comprehensive, beginner-friendly notebook that builds the Titans architecture step-by-step:

1. **Environment Setup** — GPU detection, library imports
2. **Data Pipeline** — Download and tokenize Tiny Shakespeare
3. **Model Architecture** — Build `NeuralMemory`, `TitansBlock`, `TitansGPT`
4. **MIRAS Training** — Curriculum learning with surprise weighting
5. **Analysis** — Visualize training and learned patterns
6. **Generation** — Interactive text generation

**Key Features:**
- 15+ educational visualizations
- Step-by-step explanations with analogies
- Beginner-friendly code comments

### `Titans_MIRAS_Hybrid.ipynb` — Hybrid Approach 🔗

A practical tutorial that augments a pre-trained LLM with learnable memory:

1. **Neural Memory Module** — Build trainable memory with `memorize()` and `recall()`
2. **Hybrid Engine** — Connect memory to GPT-2's hidden states
3. **Semantic Memory** — Use embeddings for fact retrieval
4. **Total Recall Experiment** — Prove memory works across context boundaries
5. **Multi-User Demo** — One LLM serving multiple users with private memories

---

## 🛠️ Installation

### Option 1: Google Colab (Recommended for Beginners)

Just click the Colab badges above! The notebooks include setup cells that install all dependencies automatically.

### Option 2: Local Installation

**Requirements:**
- Python 3.10+
- CUDA-capable GPU (recommended for Scratch notebook)

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Conda Environment (For GPU Users)

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

---

## ⚠️ Hardware Notes

These notebooks were developed on **NVIDIA DGX Spark** but are designed to run on various hardware:

| Environment | Scratch Notebook | Hybrid Notebook |
|-------------|-----------------|-----------------|
| **Google Colab (Free)** | ✅ Works (T4 GPU) | ✅ Works |
| **Google Colab Pro** | ✅ Recommended | ✅ Works |
| **Local GPU (8GB+)** | ✅ Works | ✅ Works |
| **CPU Only** | ⚠️ Slow (~30 min) | ✅ Works |

> **Tip**: For the best experience with the Scratch notebook, use a GPU with at least 8GB VRAM.

---

## 📚 References

| Resource | Link |
|----------|------|
| **Original Paper** | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) |
| **Google Research Blog** | [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) |
| **Video Explanation** | [YouTube: Titans + MIRAS Explained](https://www.youtube.com/watch?v=_WFgtK6K01g) |

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Google Research for the Titans and MIRAS papers
- The PyTorch team for the deep learning framework
- Andrej Karpathy for educational GPT implementations

---

<p align="center">
  <i>⭐ Star this repo if you find it helpful!</i>
</p>

<p align="center">
  <sub>Disclaimer: This is an educational implementation inspired by the Titans paper. It is not the official Google implementation.</sub>
</p>