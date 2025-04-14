# NeuronalDynamics

This repository contains experimental notebooks and scripts created as part of a student research project focused on decoding cognitive states from intracranial EEG (iEEG) data using transformer-based embeddings.

These experiments were developed and tested within the scope of a master's thesis at ITMO University (2025) by G. S. Boykov under the supervision of G. A. Soghoyan.

---

## 🧠 Project Context

The goal of this work is to analyze high-dimensional intracranial EEG signals recorded during memory recall tasks using modern dimensionality reduction techniques and latent space embeddings. Transformer-based models are used solely as embedding generators and are not part of this repository. They can be pre-trained to manage the biological individuality of the brain between individuals

---

## 📁 Repository Contents

- `*.ipynb`: Jupyter notebooks for preprocessing, segmentation, embedding analysis, and visualization.
- `.py`: Auxiliary scripts to run experiments as a pipeline
- `README.md`: This file.
- `environment.yml`: Optional conda environment (see below).

---

## 🚀 Model Assumptions

This repository assumes access to a pretrained embedding model capable of converting preprocessed iEEG segments into sequences of embeddings.

Expected format of embedding machine:
- Input: STFT or similarly preprocessed time-frequency iEEG data
- Output: A tensor of shape (T x D), where T = time bins, D = embedding dimension (e.g., 768)

The embedding model itself is **not included** in this repository. Any model that satisfies this input-output format can be used.

---
## 🔍 External Models

For embedding generation, any compatible transformer-based model may be used. 
In our experiments, we tested with models inspired by architectures such as [BrainBERT](https://github.com/czlwang/BrainBERT), 
which can output dense latent representations from time-frequency iEEG inputs. 

Please note: The referenced repository is used here solely for academic exploration purposes. No code from the repository is redistributed.
---

## ⚙️ Conda Setup

If you use conda, you can recreate the environment using the provided file:

```bash
conda env create -f environment.yml
conda activate brainbert
