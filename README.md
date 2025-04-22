# DL_Mini_Project2

# Parameter‑Efficient RoBERTa Fine‑Tuning with LoRA on AG News

*Satya Deep Dasari* (gd2576@nyu.edu)  
*Navdeep Mugathihalli Kumaregowda* (nm4686@nyu.edu)  
New York University — CSCI‑GA Deep Learning Spring 2025

🔗 *GitHub:* https://github.com/Navdeepmk1999/DL_Mini_Project2  
📄 *Report:* [DLProject2 (3).pdf](./DLProject2%20(3).pdf)  
📓 *Notebook:* [dl-p2.ipynb](./dl-p2.ipynb)

---

## 🚀 Project Goal

- Fine‑tune a *frozen RoBERTa‑base* model on *AG News* by training only *LoRA adapters*  
- Limit *trainable parameters* to *< 1 million* (888 580 total)  
- Achieve *≥ 84.6 %* test accuracy—within 0.4 pp of full fine‑tuning  
- Shrink checkpoint size from 476 MB → 3 MB for easy sharing and on‑device use  

---

## 🗂️ Dataset

- *AG News* (Hugging Face):  
  - 120 000 train articles, 7 600 test articles  
  - 4 balanced classes: World, Sports, Business, Sci/Tech  
- Tokenization via roberta-base tokenizer (max 512 tokens)  

---

## 🏗️ Model & LoRA Setup

| Component            | Configuration                  |
|----------------------|--------------------------------|
| *Backbone*         | roberta-base, 12 layers, 125 537 288 params (frozen) |
| *LoRA rank (r)*    | 8                              |
| *LoRA α*           | 16                             |
| *LoRA dropout*     | 0.1                            |
| *Target modules*   | query, value projections   |
| *Trainable params* | 880 896 (adapters) + 7 684 (head) = 888 580 (0.7 %) |
| *Checkpoint size*  | ≈ 3 MB                          |

---

## 🛠️ Training Configuration

- *Epochs:* 3  
- *Optimizer:* AdamW (lr = 5 × 10⁻⁵, weight decay = 0.01)  
- *Scheduler:* 10 % linear warm‑up → cosine decay  
- *Loss:* Cross‑entropy + label smoothing (ε = 0.1)  
- *Batching:* physical 16, grad‑accum 2 (eff. 32)  
- *Precision:* FP16 (PyTorch AMP)  
- *Early stopping:* best validation checkpoint retained  

---

## 📈 Results

| Metric                 | Value            |
|------------------------|------------------|
| *Test accuracy*      | 84.6 %           |
| *Trainable parameters* | 888 580         |
| *Total parameters*   | 125 537 288      |
| *Checkpoint size*    | ≈ 3 MB           |
| *Training time*      | ≈ 25 min (RTX‑3060) |

![Accuracy per Epoch](./figures/accuracy.png)  
Figure: Validation accuracy rises from 90.8 % → 92.3 % over 4 epochs.

![Loss per Epoch](./figures/loss.png)  
Figure: Training loss drops from 1.39 → 0.20, evaluation loss from 0.27 → 0.22.

![Confusion Matrix](./figures/conf_mat.png)  
Figure: Most confusion between Business ↔ Sci/Tech; overall recall 90–97 %.

![Classification Report](./figures/classification_report.png)  
Figure: Precision/recall/F1 by class; macro‑avg F1 = 0.92.

---

## 📦 Usage

1. *Clone & install*  
   ```bash
   git clone https://github.com/Navdeepmk1999/DL_Mini_Project2.git
   cd DL_Mini_Project2
   pip install -r requirements.txt
