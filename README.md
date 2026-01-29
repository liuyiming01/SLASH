# Slash the Sink: Sharpening Structural Attention Inside LLMs

This is the official PyTorch implementation for the paper **"Slash the Sink: Sharpening Structural Attention Inside LLMs"**.

---

Large Language Models (LLMs) spontaneously reconstruct graph topology internally, but this structural understanding is often diluted by the "attention sink". Our work introduces **SLASH (Structural Attention Sharpening)**, a training-free, plug-and-play solution that significantly enhances LLMs' graph reasoning capabilities by amplifying this latent structural signal at inference time.

## Key Features

*   **Training-Free Enhancement**: Improves LLM graph reasoning without any fine-tuning or architectural changes.
*   **Plug-and-Play**: Easily integrated into existing LLM inference pipelines as a lightweight module.
*   **Mechanistically Grounded**: Based on a novel theoretical analysis of the *anisotropy-isotropy conflict* within LLMs when processing serialized graphs.

## Installation

```bash
pip install -r requirements.txt
```

## Run SLASH

### 1. Offline Phase: Head Identification & Calibration

#### GraphInstruct
```bash
bash scripts/graphwiz/run_select.sh
```
```bash
bash baselines/GraphWiz/cal.sh
```
#### MolecularNet
```bash
bash scripts/molecularNet/run_select.sh
```
```bash
bash baselines/molecularNet/cal.sh
```
### 2. Online Phase: Evaluation on Benchmarks

#### GraphInstruct

```bash
bash baselines/GraphWiz/eval.sh
```

#### MolecularNet

```bash
bash baselines/molecularNet/eval.sh
```