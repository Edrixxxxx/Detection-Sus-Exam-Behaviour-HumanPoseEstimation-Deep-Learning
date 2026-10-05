# System Architectural Diagrams & Mermaid Specifications

## Project Information
- **Project Title:** Detection of Suspicious Examination Behaviours Using Human Pose Estimation and Deep Learning
- **Institution:** DMMMSU - South La Union Campus | BS Computer Science (AY 2025-2026)
- **Deployment Platform:** Desktop Application (PyQt6 with GPU-accelerated PyTorch/CUDA backend)
- **Source Code Directory:** `1_Source_Code/`
- **Diagrams Directory:** `2_Documentation/charts_and_graphs/mermaid_diagrams/`

---

## Overview

This directory contains the complete collection of formal software engineering diagrams expressed in native **Mermaid.js** syntax for the Suspicious Examination Behaviour Detection System. These diagrams capture the structural, behavioral, data, and procedural aspects of the software codebase and its academic specifications:

| Diagram Suite | Description | Documentation File | Standalone Code |
|---|---|---|---|
| **DFD (Data Flow Diagram)** | Models data propagation across external entities, processes, and persistent data stores at Level 0 (Context), Level 1 (Major Processes), and Level 2 (Sub-Processes). | [`DFD.md`](./DFD.md) | [`DFD_level0.mmd`](./DFD_level0.mmd)<br>[`DFD_level1.mmd`](./DFD_level1.mmd)<br>[`DFD_level2.mmd`](./DFD_level2.mmd) |
| **ERD (Entity Relationship Diagram)** | Models relational data architecture, entities, primary/foreign key constraints, attributes, and 1-to-many cardinalities. | [`ERD.md`](./ERD.md) | [`ERD.mmd`](./ERD.mmd) |
| **HIPO (Hierarchy Plus IPO)** | Models top-down functional decomposition via Visual Table of Contents (VTOC) and detailed Input-Process-Output subsystem flows. | [`HIPO.md`](./HIPO.md) | [`HIPO_VTOC.mmd`](./HIPO_VTOC.mmd)<br>[`HIPO_IPO.mmd`](./HIPO_IPO.mmd) |
| **Structured Chart** | Models program module execution hierarchy, multi-threading architecture, parameter passing (Data Couples), and control signals (Control Couples). | [`Structured_Chart.md`](./Structured_Chart.md) | [`Structured_Chart.mmd`](./Structured_Chart.mmd) |

---

## How to Render and Preview the Diagrams

All `.md` and `.mmd` files in this folder are formatted in standard Mermaid syntax. You can view or export them using:

1. **GitHub / GitLab Markdown Preview:**
   GitHub natively renders ````mermaid ... ```` code blocks directly in your browser.
2. **VS Code / Cursor Extensions:**
   - *Markdown Preview Mermaid Support* (by Matt Bierner)
   - *Mermaid Previewer* / *Mermaid Chart*
3. **Mermaid Live Editor:**
   Copy the content of any `.mmd` file and paste it into [mermaid.live](https://mermaid.live).
4. **Command Line (Mermaid CLI):**
   Render PNG, SVG, or PDF diagrams using `@mermaid-js/mermaid-cli`:
   ```bash
   # Install CLI
   npm install -g @mermaid-js/mermaid-cli

   # Generate SVG diagrams
   mmdc -i DFD_level0.mmd -o DFD_level0.svg
   mmdc -i DFD_level1.mmd -o DFD_level1.svg
   mmdc -i DFD_level2.mmd -o DFD_level2.svg
   mmdc -i ERD.mmd -o ERD.svg
   mmdc -i HIPO_VTOC.mmd -o HIPO_VTOC.svg
   mmdc -i Structured_Chart.mmd -o Structured_Chart.svg
   ```

---

## Synthesis with `1_Source_Code` Architecture

The diagrams are directly grounded in the implemented Python modules within `1_Source_Code`:

```
1_Source_Code/
├── main.py                     # Root CLI / GUI launcher entrypoint
├── app_gui.py                  # PyQt6 GUI: VideoThread, Main Window, Alert Tables, HUD, Snapshots
├── tracker.py                  # PoseTrackerManager: YOLOv26s-pose + ByteTrack + TrackedPerson state buffers
├── model.py                    # PyTorch ExamBehaviorLSTM neural network architecture
├── model_adapter.py            # ModelRegistry & Adapters: LSTM, Heuristic, Hybrid Ensemble, Platt Scaling
├── inference.py                # Standalone inference loop, skeleton visualizer, HUD drawing
├── config.py                   # Global system constants, paths, thresholds, class mappings
├── dataset.py                  # COCO 17-keypoint normalization and PyTorch Dataset
├── extract_dataset.py          # Dataset extraction from raw MP4 video clips into .npz feature matrices
├── train.py                    # Multi-epoch PyTorch training pipeline with early stopping & metrics
├── clipper.py                  # Annotation GUI for labeling and clipping exam behavior segments
└── Main/
    ├── main.py                 # Secondary launcher script
    └── main.md                 # UI user guide and launch instructions
```
