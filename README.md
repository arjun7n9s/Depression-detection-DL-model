<div align="center">

# 🧠 MindSense
### **Multimodal Depression Detection AI**

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br>

*An advanced multi-task, quality-aware deep learning architecture designed for real-time behavioral screening and depression risk estimation using high-dimensional facial and acoustic modalities.*

<br>

**[ System Architecture ](#-system-architecture) • [ Key Features ](#-key-features) • [ Implementation Roadmap ](#-implementation--roadmap) • [ Ethics & Privacy ](#%E2%9A%96%EF%B8%8F-ethics--privacy)**

</div>

---

## 📖 Overview

MindSense is an ongoing implementation of a multimodal deep learning system designed to estimate depression risk via non-invasive behavioral signals. By leveraging high-dimensional facial expressions (landmarks, pose, gaze) and acoustic features, this predictive pipeline delivers real-time inference suitable for clinical screening augmentation.

### 📊 Integrated Datasets
*   **E-DAIC:** Highly controlled clinical interviews *(PHQ-8 Continuous + Binary labels)*.
*   **D-Vlog:** "In-the-wild" YouTube vlogs highlighting continuous speech and behavior *(Binary labels)*.

> [!WARNING]  
> **Clinical Disclaimer:** This tool acts strictly as a **behavioral screening support system** and is not a clinical diagnostic instrument.

<br>

## 🏗 System Architecture

The core of the system relies on a **Source-Aware Multimodal Transformer** that applies quality-aware modality gating before fusing features through cross-modal attention.

```mermaid
graph TD
    classDef visual fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef acoustic fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    %% Data Ingestion
    V_Raw["🎬 Visual Stream <br> (T, V_dim)"]:::visual
    A_Raw["🎙️ Acoustic Stream <br> (T, A_dim)"]:::acoustic

    %% Projection
    V_Proj["Source-Specific Projection <br> 128-dim"]:::visual
    A_Proj["Source-Specific Projection <br> 128-dim"]:::acoustic
    
    V_Raw --> V_Proj
    A_Raw --> A_Proj

    %% Encoding
    V_Enc["Visual Encoder <br> 1D CNN + BiGRU"]:::visual
    A_Enc["Acoustic Encoder <br> TCN + BiGRU"]:::acoustic

    V_Proj --> V_Enc
    A_Proj --> A_Enc

    %% Gating
    V_Gate["Modality Gate <br> Quality-aware Gating"]:::visual
    A_Gate["Modality Gate <br> Quality-aware Gating"]:::acoustic

    V_Enc --> V_Gate
    A_Enc --> A_Gate

    %% Fusion
    F_Cross["Cross-Modal Attention Fusion <br> 4-heads + Masking Logic"]:::core

    V_Gate --> F_Cross
    A_Gate --> F_Cross

    %% Outputs
    HeadA(("E-DAIC <br> Binary Risk")):::output
    HeadB(("E-DAIC <br> PHQ Regression")):::output
    HeadC(("D-Vlog <br> Binary Risk")):::output

    F_Cross --> HeadA
    F_Cross --> HeadB
    F_Cross --> HeadC
```

<br>

## 🚀 Key Features

### 🧩 Robust Multimodal Architecture
*   **Missing Modality Masking:** Seamlessly ingests unimodal inputs when either facial tracking or audio streams drop below minimum confidence thresholds in real-time.
*   **Quality-Aware Gating:** Automatically down-weights unreliable streams via dynamically computed confidence scores (e.g., MediaPipe tracker uncertainty, poor VAD confidence).

### 🎯 Advanced Multi-Task Learning Strategy
*   **Joint Optimization:** Co-trains both datasets using a unique Multi-Task Loss methodology targeting regression and binary thresholds simultaneously: 
    *   `L = α(L_binary_edaic) + β(L_phq_regression) + γ(L_binary_dvlog)`
*   **Focal Loss Tuning:** Custom focal loss down-weights overwhelmingly clear samples, natively adapting to inherent dataset class imbalances.

### ⚡ Sub-Second Real-Time Inference
*   **Live Overlays:** Smooth, 60 FPS feature tracking overlays mapped to user webcam feeds.
*   **Async Execution:** Heavy inference operations are fully decoupled from tracking loops to prevent UI blocking.
*   **Bridge Learning:** Employs feature-bridge models allowing weights trained on heavy offline extractors (OpenFace/eGeMAPS) to accept lightweight, localized tracker outputs in production (MediaPipe/Librosa).

<br>

## 🛠️ Implementation & Roadmap

<details open>
<summary><b>Phase 1 & 2: Dataset Verification & Ingestion</b></summary>
<br>

- ✅ Unify E-DAIC and D-Vlog data representations.
- ✅ Implement quality constraints on OpenFace parameters.
- ✅ Build dynamic manifest generators with strict sequence validation logic.
</details>

<details open>
<summary><b>Phase 3 & 4: Training & Modelling</b></summary>
<br>

- ✅ Verify baseline ladder metrics (Acoustic vs. Visual unimodality performance bounds).
- ✅ Construct source-conditioned LayerNorm fusion architectures to bridge dataset domains.
- ✅ Formulate temporal subject-level prediction aggregators.
</details>

<details open>
<summary><b>Phase 5 & 6: Deployment & Feature Matching</b></summary>
<br>

- ✅ Develop MediaPipe-to-OpenFace Feature Bridge projection prototypes.
- ⏳ Establish Flask / FastAPI async socket communication pipelines.
- ⏳ Integrate Text modality transcript parsing + Rule-based NLP sentiment gating.
- ⏳ Launch dashboard enhancements for real-time visualization.
</details>

<br>

## 📂 Repository Access

> [!NOTE]  
> The underlying heavy binaries (`.mp4`, `.npy`, and `.mat` datasets) are intentionally `.gitignored` to comply with data privacy policies and LFS storage constraints. Only core logic, models, architectures, and implementation plans are exposed.

```mermaid
graph LR
    Root[📂 Depression-detection-DL-model]
    Html[📄 Data analysis and plan.html]
    Plan[📄 implementation_plan.md]
    Src[📁 src/]
    D[📁 data/]
    M[📁 model/]
    I[📁 inference/]

    Root --> Html
    Root --> Plan
    Root --> Src
    Src --> D
    Src --> M
    Src --> I
    
    style Root fill:#f1f8ff,stroke:#0366d6
    style Src fill:#fff5f5,stroke:#cb2431
```

<br>

## ⚖️ Ethics & Privacy 

**Bias Mitigation Strategy:**
This model completely avoids text transcription features (unless heavily scrubbed) to bypass severe "Interviewer Prompts Bias" inherent to clinical datasets. Demographic bias tracking isolates variance across *(Male vs. Female)* and *(Age brackets)* ensuring balanced screening integrity. 

**Data Locality:**
Tracking captures are localized telemetry bounds. The system runs entirely inference-edge using memory buffers; absolutely zero raw video or audio frames are recorded, stored, or forwarded over network limits.
