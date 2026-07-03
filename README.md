# Neural vs. Classical Decoders for the Surface Code: From Simulation to IBM Quantum Hardware

---

## Introduction

Quantum computers require logical error rates near \(10^{-12}\) for practical applications, yet physical qubits exhibit errors at 0.1–1%. This project directly addresses this gap by evaluating data-driven deep learning architectures against the classical industry standard, Minimum-Weight Perfect Matching (MWPM).

Using a distance-3 rotated surface code layout, we benchmark MWPM against a Convolutional Neural Network (CNN), a standard Transformer trained from scratch, and a Hybrid Transformer trained using an advanced two-stage pipeline. We demonstrate that a two-stage training scheme—pre-training a model on clean symmetric noise to learn core lattice topology, then fine-tuning on complex asymmetric noise—creates a resilient decoder capable of overcoming hardware biases better than traditional mathematical graph algorithms on both simulation environments and actual physical quantum hardware.

---

## Folder Structure

```
AlphaQubit-Neural-Decoding/
│
├── README.md
├── requirements.txt
│
├── 01_Data_Generation_Simulated.ipynb
├── 02_Data_Generation_IBM_Hardware.py
├── 03_MWPM_and_Transformer_Training.ipynb
├── 04_CNN_Training.ipynb
├── 05_Evaluation_Simulated.ipynb
├── 06_Evaluation_Real_Hardware.ipynb
│
├── data/          # Generated features and labels (.csv)
├── models/        # Trained network checkpoints (.pth)
└── results/       # Pipeline plots and final logs
```

---

## How to Run This Code

### 1. Environment Setup

Install the necessary package ecosystem using the dependency constraints file:

```bash
pip install -r requirements.txt
```

Libraries installed: qiskit>=1.0.0, qiskit_aer, qiskit-ibm-runtime, pymatching, torch, pandas, numpy, matplotlib, scikit-learn

### 2. IBM Quantum Setup

To run the live processor data pipeline script (`02_Data_Generation_IBM_Hardware.py`), provide your cloud computing authorization keys:

- Open the script file and locate the configuration variables at the top
- Paste your personal access token into `IBM_TOKEN = "YOUR_TOKEN_HERE"`
- Provide your cloud resource instance name in the respective parameter fields

### 3. Execution Sequence

The code assets must be processed sequentially from notebook 01 to 06.

**Manual Active Folders:**
- Generated `.csv` output files from data generators 01 and 02 must be kept in or moved to the `/data/` folder so downstream neural networks in 03 and 04 can ingest them
- Ensure trained PyTorch `.pth` matrices remain saved in the `/models/` folder before running evaluation segments 05 and 06

**Note:** Simulator noise data generation runs deterministically under set seeds. Real-world hardware outputs will fluctuate slightly over separate job runs due to natural physical quantum drift.

---

## Results and Findings

Our core architecture and findings can be analyzed using the design pipeline plots found in the `/results/` folder.

### Data Generation (Noise Injection Pipeline)

We implemented a 17-physical-qubit distance-3 rotated surface code circuit structure (9 data qubits, 8 stabilizer ancillas) iterating over 5 syndrome extraction rounds.

- **Symmetric Depolarizing Noise (10,000 shots):** Theoretical base model with equal bit-flip (X) and phase-flip (Z) probabilities. Utilized for the first stage of the hybrid pipeline to map pure surface code structural constraints.
- **Asymmetric Pauli Noise (10,000 shots):** Implements physical qubit relaxation constraints with a 1.5% measurement error and a 0.5% reset error biased toward bit-flips. Drives the secondary model fine-tuning stage.
- **IBM Physical Quantum Noise (1,000 shots):** Extracted directly from the 156-qubit `ibm_marrakesh` processor via SamplerV2 primitives to pull native probability outcomes instead of basic averages.

*(See `results/noise_injection.png` for visualization of the tracking pipeline from gates to clean 40-bit readout tensors.)*

### Decoder Architecture and Evaluation Flow

Instead of calculating hardcoded classical graphs, machine learning models treat raw 40-bit syndrome strings directly to trace correlations:

- **The Baseline:** MWPM via PyMatching constructs rigid geometric error arrays, performing well under symmetric noise but dropping accuracy under asymmetric biases.
- **The Neural Shift:** The CNN maps local error patches while the Transformer's self-attention captures global lattice connections. However, training a complex model entirely from scratch under biased data bottlenecks quickly.
- **The Hybrid Transformer Blueprint:** By employing a two-stage training strategy, the model uses simulated symmetric patterns to baseline the mathematical rules of the grid before fine-tuning variables on hardware noise.

*(See `results/decoder_architecture.png` to follow the exact model evaluation pathways.)*

### Key LER Outcomes

**Simulated Asymmetric Biases:**
- Hardcoded graph connections in MWPM collapsed under uneven error injection, pushing its Logical Error Rate (LER) to 19.90%.
- The two-stage Hybrid Transformer successfully tracked the drift, achieving a superior 12.65% LER.

**Physical Device Ingestion:**
- Physical hardware crosstalk underwhelmed the generic CNN baseline, which dropped behind the classical algorithm (51.40% LER vs. MWPM's 50.50% LER).
- The Hybrid Transformer successfully translated its sim-to-hardware data workflow, securing the best real-world benchmark at 49.70% LER.

These findings show that while simple neural networks can fail on raw processor data, an organized two-stage data-driven pipeline successfully overcomes chaotic noise properties that traditional, hardcoded decoders are fundamentally unequipped to track.
