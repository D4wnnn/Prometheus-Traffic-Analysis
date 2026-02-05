# Prometheus-Traffic-Analysis

Code for the paper **Prometheus: Protocol-Aware Asymmetric Collaboration for Efficient and Robust Encrypted Traffic Classification**.

The codebase is organized into three parts:

| Component | Description |
|-----------|--------------|
| **Heavy Model** | Pre-trained + fine-tuned backbone (full packet/byte input). Used for accurate classification and as teacher for the light model. |
| **Light Model** | Small student model trained via distillation with **global byte selection**; runs on a subset of bytes for efficiency. |
| **Collaborative System** | Online deployment where **Light** handles most traffic; **Heavy** is queried only when Light is uncertain. Includes online fine-tuning of Light. |

---

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Data Processing](#data-processing)
- [Heavy Model](#heavy-model)
  - [Pre-training](#pre-training)
  - [Fine-tuning](#fine-tuning)
- [Light Model](#light-model)
  - [Global Selection](#global-selection)
  - [Distillation](#distillation)
- [Collaborative System](#collaborative-system)
- [Pipeline Summary](#pipeline-summary)

---

We provide **two pre-trained backbone checkpoints**: **10×300** (100 packets × 300 bytes; fine-tuning uses 10 packets for easier extension of packet count) and **5×320** (for constrained input). Weights are under `outputs/pretrain_checkpoints`.

---

## Quick Start

We provide a **pre-processed AES-128-GCM H5 dataset**. With it you can run Heavy model fine-tuning directly without doing data processing or pre-training.

1. **Install dependencies** (see [Environment Setup](#environment-setup)):
   ```bash
   pip install -r requirements.txt
   ```

2. **AES-128-GCM H5 data** is already included in `pacp2h5/aes_128_gcm_h5`.

3. **Run fine-tuning** (single GPU example):
   ```bash
   cd Prometheus-Traffic-Analysis/finetune

   python finetune.py \
       --data_dir       ../pacp2h5/aes_128_gcm_h5 \
       --pretrain_path  ../outputs/pretrain_checkpoints/10x300/10x300_pretrain.pth \
       --config_path    ../outputs/pretrain_checkpoints/10x300/backbone_config.py \
       --save_path      ../outputs/finetuned_best/aes_128_gcm/finetuned_best_aes_128_gcm.pth \
       --batch_size     128 \
       --epochs         20 \
       --lr             1e-3 \
       --warmup_epochs  10 \
       --gpus           0
   ```

4. **Multi-GPU:** Set `--gpus 0,1,2,3` (or your available GPU indices) to use multiple GPUs.

After training, the fine-tuned Heavy model is saved at `--save_path`. You can then run evaluation (see `eval/`) or continue with [Light Model](#light-model) (selection + distillation) and the [Collaborative System](#collaborative-system).

---

## Environment Setup

### Requirements

- Python 3.11
- CUDA 12.x (for GPU training)
- conda or venv recommended for environment management

### Installation

```bash
# Clone the repository
git clone <repo_url>
cd Prometheus-Traffic-Analysis

# Create virtual environment (optional)
conda create -n prometheus python=3.11
conda activate prometheus

# Install dependencies
pip install -r requirements.txt
```

---

## Data Processing

Data processing has three stages: **raw PCAP → flow-level split** → **train/val/test split** → **convert to H5** for Heavy/Light training and evaluation.

### 1. PCAP Flow-Level Processing (data-preprocess/pcap-process)

Takes raw PCAPs organized by category and produces one-PCAP-per-flow directories for feature extraction.

**Input directory layout (example):**

```
input/
├── CategoryA/
│   ├── xxx.pcap
│   └── ...
├── CategoryB/
│   └── ...
```

**Usage:**

```bash
cd data-preprocess/pcap-process

python main.py \
    --input   /path/to/your/raw_pcap_by_category \
    --output  /path/to/processed_output \
    --clear \
    --min-category-flows 15
```

**Common arguments:**

| Argument | Description |
|----------|-------------|
| `--input` | Root directory of PCAPs organized by category |
| `--output` | Output directory (train/val/test will be created here) |
| `--clear` | Clear output directory if non-empty |
| `--min-category-flows` | Minimum flows per category; categories below this are dropped |
| `--train-ratio` / `--val-ratio` / `--test-ratio` | Split ratios (optional) |
| `--filter-rule` | Packet filter rule (optional, e.g. TCP/UDP only) |

For batch processing multiple datasets, edit paths and commands in `run_all.sh`.

### 2. Train/Val/Test Split (data-preprocess/process_aes)

For data already organized by category (one folder per class) but not yet split into train/val/test (e.g. AES-style encrypted datasets).

**Input:** One subdirectory per class, each containing `.pcap` files.  
**Output:** Under `output`, creates `train/<class>/`, `val/<class>/`, `test/<class>/`.

```bash
cd data-preprocess/process_aes

python split_dataset.py \
    --input  /path/to/initial_datasets/finetune/aes-128-gcm \
    --output /path/to/processed_datasets/finetune/AES-128-GCM-processed \
    --train  0.7 \
    --val    0.15 \
    --test   0.15
```

For multiple datasets (e.g. aes-128-gcm, aes-256-gcm, chacha20-poly1305, mix), use `run_aes.sh` as reference.

### 3. PCAP → H5 (pacp2h5)

Converts processed PCAPs (organized as train/val/test + class) into HDF5 for pre-training, fine-tuning, and Light model training.

#### 3.1 Pre-training Data (unlabeled)

Pre-training data is **unlabeled**; the H5 file contains only multi-modal features (bytes, sequence, stats, etc.).

- **Script:** `pacp2h5/preprocess_pretrain_data.py`
- **Config:** Edit `dataset_dirs` (list of pre-training dataset directories) and `output_h5_file` (output H5 path) inside the script.
- **Run:**  
  ```bash
  cd pacp2h5
  python preprocess_pretrain_data.py
  ```
  Or use the wrapper: `./1-get_pretrain_data.sh`

**Output:** A single `.h5` file with datasets such as `bytes_nlp_view`, `seq_nlp_view`, `tcp_data_view`, `stats_view`, `num_nodes`.

#### 3.2 Fine-tuning Data (labeled)

Fine-tuning data is **labeled**; the H5 includes the same features plus `labels`. Used for both Heavy fine-tuning and Light distillation.

- **Script:** `pacp2h5/preprocess_finetune_data.py`
- **Input:** Per-dataset directory with `train/<class>/`, `val/<class>/`, `test/<class>/` containing `.pcap` files.
- **Example:**  
  ```bash
  cd pacp2h5

  python preprocess_finetune_data.py \
      --input  /path/to/finetune/CSTNET \
      --output ./cstnet_h5

  python preprocess_finetune_data.py \
      --input  /path/to/finetune/DataCon2021-processed/part1 \
      --output ./datacon2021_part1_h5
  ```
  For multiple datasets, use `2-get_finetune_data.sh` and set `BASE_DIR` to your dataset root.

**Output:** One directory per dataset (e.g. `cstnet_h5/`) with H5 files and labels, used as `--data_dir` in Heavy fine-tuning and Light scripts.

---

## Heavy Model

The Heavy model is the full backbone: pre-trained on unlabeled traffic, then fine-tuned on labeled data. It serves as the **teacher** for the Light model and as the **expert** in the collaborative system when the Light model is uncertain.

### Pre-training

Runs self-supervised pre-training (e.g. masking) on unlabeled H5 data to obtain backbone weights and a matching `backbone_config.py`.

**Example:**

```bash
cd scripts

OUTPUT_DIR=../outputs/pretrain_checkpoints/10x300
mkdir -p $OUTPUT_DIR

python ../pretrain/pretrain.py \
    --data_file   /path/to/your/pretrain_data.h5 \
    --batch_size  16 \
    --epochs      20 \
    --lr          1e-4 \
    --mask_prob   0.25 \
    --save_dir    $OUTPUT_DIR \
    --gpus        0,1,2,3,4,5,6,7
```

Or use the provided script (edit `--data_file` to your H5 path first):

```bash
bash scripts/1-run_pretrain.sh
```

**Key arguments:** `--data_file`, `--save_dir`, `--batch_size`, `--epochs`, `--lr`, `--mask_prob`, `--resume` / `--resume_path`, `--gpus`.

### Fine-tuning

Fine-tunes the pre-trained backbone on labeled H5 data for downstream classification. Produces the **teacher / Heavy** checkpoint used by Light and the system.

**Single-dataset example:**

```bash
python ../finetune/finetune.py \
    --data_dir      /path/to/cstnet_h5 \
    --pretrain_path ../outputs/pretrain_checkpoints/5x320/5x320_pretrain_epoch_7.pth \
    --config_path   ../outputs/pretrain_checkpoints/5x320/backbone_config.py \
    --batch_size    16 \
    --epochs        20 \
    --lr            1e-3 \
    --warmup_epochs 10 \
    --freeze_mode   none \
    --save_path     ../outputs/finetuned_best/cstnet/finetuned_best_cstnet.pth \
    --gpus          0,1,2,3
```

**Batch fine-tuning (multiple datasets):** Set `DATA_BASE_DIR`, `PRETRAIN_CHECKPOINTS`, and `config_path` in the script, then:

```bash
bash scripts/2-run_finetune.sh
```

The script runs fine-tuning on CSTNET, DataCon2021-Part1, AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305, and Mix. Data directories are expected as `$DATA_BASE_DIR/{cstnet,datacon2021_part1,...}_h5`.

**Key arguments:** `--data_dir`, `--pretrain_path`, `--config_path`, `--save_path`, `--freeze_mode`, `--gpus`. If `--pretrain_path` is omitted, training starts from scratch; you still need `--config_path` for the backbone architecture.

---

## Light Model

The Light model is a small student that consumes only a **selected subset of bytes** per packet (from global importance selection). It is trained by **knowledge distillation** from the fine-tuned Heavy (teacher). Pipeline: **build global selectors** → **train Light with distillation**.

### Global Selection

Builds per-dataset **global byte selectors** using the Heavy (teacher) model: which bytes/packets are most important for the teacher’s predictions. The output `.npz` is used by the Light training script.

**Script:** `distillation/build_global_importance.py`  
**Runner:** `scripts/4-run_selection.sh`

Configure `DATA_BASE_DIR`, `TEACHER_BASE_DIR`, and `OUT_DIR` in the script. Teacher checkpoints are expected under `$TEACHER_BASE_DIR/<dataset>/finetuned_best_<dataset>_swa.pth`.

```bash
cd scripts
bash 4-run_selection.sh
```

This runs selection for CSTNET, AES-128/256-GCM, ChaCha20-Poly1305, Mix, and DataCon2021-Part1. Selectors are saved under `$OUT_DIR` (e.g. `cstnet_per_packet_5x3x100.npz`). Parameters such as `TOP_PACKETS`, `WINDOW_LEN`, `WINDOWS_PER_PACKET`, and `byte_strategy` (e.g. `per_packet`, `contiguous`) can be adjusted in the script.

### Distillation

Trains the Light (student) model on the **selected bytes** with knowledge distillation from the Heavy (teacher). Uses the global selector `.npz` from the selection step.

**Script:** `distillation/train_light_offline_ddp_select.py`  
**Runner:** `scripts/5-distillation_select.sh`

Set `DATA_BASE_DIR`, `TEACHER_BASE_DIR`, `SELECTOR_DIR`, and `OUTPUT_BASE_DIR`. The script expects:

- Teacher: `$TEACHER_BASE_DIR/<dataset>/finetuned_best_<dataset>_swa.pth`
- Selector: `$SELECTOR_DIR/<dataset>_<byte_strategy>_<select_config>.npz` (e.g. `5x3x100` must match the selection step)

```bash
cd scripts
bash 5-distillation_select.sh
```

Training parameters (e.g. `BATCH_SIZE`, `EPOCHS`, `LR`, `ALPHA`, `TEMPERATURE`, `--student_aug`) can be edited in the script. Light checkpoints are saved under `$OUTPUT_BASE_DIR/<dataset>_global_<strategy>_<config>_distilled/`.

---

## Collaborative System

The **collaborative system** deploys Light and Heavy together: the **frontend** runs the Light model and optionally queries the **backend** (Heavy) when confidence is below a threshold; an **online trainer** can update the Light model on the fly. Code and experiments live under **`system_online_h5_format/`**.

### Layout

| Directory | Purpose |
|-----------|---------|
| `exp1-online-learning` | Online learning experiments (e.g. source→target drift, number of drift epochs). |
| `exp2-six-datasets` | Six-dataset evaluation: **Pure Light** baseline vs **System** (Light + Heavy + Online Trainer). |
| `exp3-hyperparameter` | Hyperparameter experiments for the system. |
| `exp4-embedding_save` | Experiments with embedding save/analysis. |

Each experiment folder contains:

- **`backend_server.py`** — Heavy model server (loads fine-tuned Heavy, runs inference, can send samples to trainer).
- **`frontend_client.py`** — Client that runs Light (with global selector), optionally calls Heavy when below `conf_threshold`, and drives evaluation.
- **`online_trainer.py`** — Updates the Light model online using data forwarded from the backend.
- **`run_experiments.py`** — Orchestrates processes (backend, trainer, frontend) and runs the experiment (e.g. Pure Light vs System per dataset).

### Running system experiments

1. Ensure Heavy and Light checkpoints and global selectors exist (Heavy fine-tuning + scripts 4 and 5).
2. In the desired experiment folder (e.g. `exp2-six-datasets`), set paths in `run_experiments.py`: `PATH_PREFIX`, `PRETRAIN_CHECKPOINT`, dataset paths (e.g. `heavy_path`, `light_path`, `selector_path` from `get_dataset_config`).
3. Run:

   ```bash
   cd system_online_h5_format/exp2-six-datasets
   python run_experiments.py
   ```

Experiments typically start backend and trainer as background processes, then run the frontend client in **pure** (Light only) or **system** (Light + Heavy + online trainer) mode and write results under `results/`.

---

## Pipeline Summary

1. **Environment:** `pip install -r requirements.txt`; set up CUDA/PyTorch as needed.
2. **Data:**  
   - Raw PCAP → `data-preprocess/pcap-process` (flow-level split).  
   - Optional split → `process_aes/split_dataset.py`.  
   - Pre-training: `pacp2h5/preprocess_pretrain_data.py` → one unlabeled H5.  
   - Fine-tuning / Light: `pacp2h5/preprocess_finetune_data.py` → one H5 directory per dataset.
3. **Heavy model:**  
   - Pre-train: `pretrain/pretrain.py` → checkpoints + `backbone_config.py`.  
   - Fine-tune: `finetune/finetune.py` → Heavy (teacher) checkpoints per dataset.
4. **Light model:**  
   - Selection: `scripts/4-run_selection.sh` → global selector `.npz` per dataset.  
   - Distillation: `scripts/5-distillation_select.sh` → Light checkpoints per dataset.
5. **Collaborative system:**  
   - Run experiments in `system_online_h5_format/exp*` (e.g. `run_experiments.py`) using the same Heavy/Light/selector paths.

For evaluation-only scripts (Heavy-only, Light-only, system, robustness, etc.), see `eval/` and `scripts/` (e.g. `3-eval-only-heavy.sh`, `6-eval-only-light.sh`). For distillation utilities and alternate training scripts, see `distillation/`.
