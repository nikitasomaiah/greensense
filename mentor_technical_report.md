# GreenSense: Computer Vision Pipeline for Landscape Asset Census & Audit Automation

**Technical Internship Deliverable Report**  
**Project Name:** GreenSense — Tree & Asset Census Module  
**Domain:** Horticulture & Landscaping Division, Facilities Management  
**Target SOP:** Embassy Group Horticulture & Landscaping SOP  

---

## 1. Executive Summary & Project Rationale

In large-scale commercial facilities management (FM) contracts—such as Embassy Manyata, Embassy GolfLinks, and Embassy Boulevard—maintaining landscaping assets (trees, palms, shrubs, hedges, and flower beds) represents both a primary aesthetic mandate and a substantial financial liability. Under standard operating procedures (SOP), field supervisors perform daily site walk-throughs to verify plant growth, complete manual paper checklists (Annexure 1), and post site photos to operational WhatsApp groups.

### Operational Challenges Addressed
- **High Labor Cost & Human Error**: Manual line-item counting across multi-acre tech parks is slow, error-prone, and inconsistent across shifts.
- **Delayed Loss Detection**: Tree mortality resulting from storm damage, disease, or theft often goes unnoticed until routine quarterly reviews, creating contract SLA non-compliance and replacement cost disputes.
- **Audit Documentation Overhead**: Compiling data for half-yearly SME Landscaping Audits requires manually consolidating paper logs into spreadsheets.

### System Solution
**GreenSense** digitizes this workflow into an end-to-end computer-vision pipeline. Using ordinary site photos already collected by field supervisors, the system automatically detects and counts landscape assets, auto-detects zones via EXIF GPS metadata, logs timestamped records into an SQLite database, flags sudden asset count drops using a rolling baseline algorithm, and generates one-click CSV/Excel audit reports.

---

## 2. System Architecture & Technical Methodology

The system architecture is structured into modular, decoupled Python components:

```text
GreenSense Pipeline
├── Image Input (Single / Batch Photo Upload)
├── EXIF GPS Extraction & Zone Mapping (utils/geotag.py)
├── Open-Vocabulary Object Detection (detect.py - OWL-ViT)
├── Hybrid LLM Horticultural Suite (llm_advisor.py)
│   ├── Dynamic Vision-Language Prompt Expansion
│   ├── Automated Loss Incident Diagnostics & SOP Remedial Action Plans
│   ├── SME Landscaping Audit Executive Memorandum Generator
│   ├── Visual Plant Health & Foliage Diagnostics
│   └── GreenSense Natural Language Copilot
├── SQLite Database & Rolling Loss Alert Engine (asset_register.py)
├── Quantitative Accuracy Benchmark Engine (evaluate.py)
└── Streamlit Web Dashboard (app.py)
```

### 2.1 Open-Vocabulary Object Detection (`detect.py`)
Rather than training a closed-set detector (such as YOLO or Faster R-CNN) from scratch—which requires thousands of hand-labeled bounding box annotations—GreenSense utilizes **OWL-ViT (`google/owlvit-base-patch32`)**, an open-vocabulary vision-language model developed by Google Research.

- **Zero-Shot Multimodal Alignment**: OWL-ViT projects image patches and plain-text query strings into a shared embedding space using CLIP text-image representations.
- **SOP Category Mapping**: Detection vocabulary is pulled directly from Embassy's "Plant & Material Selection" categories (`"a tree"`, `"a palm tree"`, `"a flowering shrub"`, `"a hedge"`, `"a flower bed"`, `"a lawn / grass area"`, `"a pergola"`, `"a garden pathway"`).
- **Input Validation & EXIF Rotation**: Implements image integrity verification and automatic orientation handling using `PIL.ImageOps.exif_transpose`.
- **Version-Agnostic Post-Processing**: Includes adaptive hooks compatible with both `transformers` v4 and v5 (`post_process_grounded_object_detection`, `post_process_object_detection`, and `image_processor.post_process_object_detection`).

### 2.2 Confidence Threshold Sensitivity Analysis
Detection sensitivity is configurable via a confidence threshold parameter (\(\tau\)):
- **Low Threshold (\(\tau = 0.05 - 0.10\))**: High Recall. Detects small, distant, or heavily occluded plants at the expense of potential false positives.
- **Standard Threshold (\(\tau = 0.12 - 0.15\), Default)**: Balanced setting optimized for typical outdoor site survey photos.
- **High Threshold (\(\tau = 0.20+\))**: High Precision. Minimizes false alarms, suited for formal audit verification photos.

### 2.3 EXIF GPS Geotag Extraction (`utils/geotag.py`)
- Reads embedded EXIF GPS tags from smartphone site photos.
- Converts Degrees/Minutes/Seconds (DMS) rationals to decimal degrees (\(\text{Latitude}, \text{Longitude}\)).
- Maps coordinates to reference Embassy facility zones in Bengaluru (Manyata: `13.0455° N, 77.6201° E`, GolfLinks: `12.9602° N, 77.6484° E`, Boulevard: `13.1605° N, 77.5802° E`).
- **Graceful Degradation**: If EXIF metadata is missing or corrupted, the system falls back seamlessly to manual zone selection without throwing runtime exceptions.

### 2.4 Database Architecture & Loss Alert Engine (`asset_register.py`)
- **SQLite Storage**: Data is persisted in `asset_register.db` (`census_log` table), ensuring concurrent write safety and transaction integrity while maintaining signature compatibility with original CSV functions (`log_census`, `load_register`, `zone_history`, `check_for_loss`, `summary_report`).
- **Input Validation**: Rejects negative asset counts (\(\text{count} \ge 0\)), empty zone names, and malformed date formats with explicit `ValueError` exceptions.
- **Legacy Auto-Migration**: On initialization, existing `asset_register.csv` records are automatically validated and migrated into SQLite.
- **Rolling Baseline Loss Alerting**: Compares new count (\(C_{\text{latest}}\)) against the rolling mean (\(\mu_{\text{recent}}\)) of the last 5 readings for a specific zone and asset label. If \(C_{\text{latest}} < 0.7 \times \mu_{\text{recent}}\) (a 30%+ drop), a high-priority loss/damage alert is triggered.

### 2.5 Horticultural AI Intelligence & Hybrid LLM Suite (`llm_advisor.py`)
To elevate GreenSense from a purely numerical counting tool into an intelligent agronomic decision-support platform, we integrated a dedicated **Hybrid LLM Intelligence Suite**:

1. **Dual-Mode Hybrid Architecture**:
   - **Built-in Agronomic Expert Engine (Offline Default)**: Zero-dependency, deterministic knowledge synthesis grounded in Embassy Group Horticulture SOPs, Bengaluru red soil / STP water chemistry, and South Indian flora pathology. Guarantees 100% functionality during field walk-throughs or offline viva demonstrations without network calls or API keys.
   - **Cloud LLM API Connectivity**: Seamlessly integrates with Google Gemini (`gemini-1.5-flash`, `gemini-1.5-pro`) and OpenAI-compatible endpoints when an API key is provided, falling back automatically to the offline expert engine on network dropouts.

2. **Automated Loss Incident Diagnostics & Remedial Action Plans**:
   - Translates rolling baseline alerts (\(\ge 30\%\) drops) into structured diagnostic dossiers.
   - Generates ranked agronomic root causes (e.g., STP water TDS/salinity spikes \(>1200\,\text{ppm}\), Phytophthora root rot, underground MEP utility trenching).
   - Produces structured SOP-mandated remedial protocols across three operational time horizons:
     - **24-Hour Immediate**: Physical root-crown and moisture audit under SOP §4.2.
     - **7-Day Corrective**: Systemic fungicide/neem drench and nursery warranty replacement work-orders under SOP §5.3.
     - **30-Day Monitoring**: Weekly follow-up photo census and soil aeration.
   - Computes financial replacement liability estimates in INR based on standard commercial facility management benchmarks.

3. **Dynamic Open-Vocabulary Query Expansion**:
   - Enhances OWL-ViT's text prompt embeddings by taking natural language scene descriptions (e.g., *"tropical poolside garden with royal palms, flowering bougainvillea, and trimmed ficus hedges"*) and generating specialized detection queries (`"a tall royal palm tree"`, `"a blooming bougainvillea shrub"`).
   - Dynamically maps specialized visual labels back to canonical Embassy SOP asset classes, ensuring database consistency.

4. **SME Landscaping Audit Executive Memorandum Generator**:
   - Transforms multi-zone SQLite census tables and rolling anomaly flags into polished, executive-ready Markdown memorandums for the Facilities Management Committee and Estate Directors.

5. **GreenSense AI Landscape Copilot**:
   - An interactive conversational assistant grounded in real-time database queries, zone risk profiles, and official Embassy Horticulture SOP operational standards.

---

## 3. Empirical Accuracy Evaluation & Benchmarking

To rigorously evaluate detection accuracy, an empirical benchmark engine ([evaluate.py](file:///Users/nikita/Desktop/EMBASSY%20INTERNSHIP/evaluate.py)) was built to test `CensusDetector` against 15 ground-truth annotated scenes (`eval_dataset/`).

### 3.1 Mathematical Formulations
1. **Mean Absolute Error (MAE)**:
   \[
   \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} | y_i - \hat{y}_i |
   \]
2. **Mean Absolute Percentage Error (MAPE)**:
   \[
   \text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
   \]
3. **Count Bias Direction**:
   \[
   \text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
   \]

### 3.2 Benchmark Results Summary

| Evaluation Metric | Empirical Score | Operational Interpretation |
|---|---|---|
| **Benchmark Scenes Evaluated** | `15 Scenes` | Representative outdoor geometries and lighting conditions. |
| **Total Samples Evaluated** | `46 Categories` | Evaluated across trees, palms, shrubs, hedges, and lawn areas. |
| **Mean Absolute Error (MAE)** | `6.15 assets/photo` | Average absolute numerical count error per site photo. |
| **Mean Pct Error (MAPE)** | `84.2%` | Relative percentage deviation across variable plant sizes. |
| **Count Bias Direction** | `-5.28` | Systematic negative bias (undercounting due to canopy overlap). |

### 3.3 Critical Technical Findings & Engineering Justification
The empirical finding of a negative count bias (\(-5.28\)) demonstrates that single-image open-vocabulary detection systematically undercounts plants when foliage overlaps (canopy occlusion). 

Because single-photo count precision is physically limited by 2D camera perspectives, relying on a single absolute count is insufficient for auditing. This empirical result directly justifies our engineering design choice to implement **rolling average baseline loss alerting**, which evaluates relative percentage drops over time rather than relying on single-snapshot count perfection.

---

## 4. Web Dashboard & System Interface (`app.py`)

The Streamlit front-end ([app.py](file:///Users/nikita/Desktop/EMBASSY%20INTERNSHIP/app.py)) provides a clean, professional user interface for field supervisors and FM managers, organized into five functional tabs:

1. **New Census Audit Tab**:
   - Supports **Single Site Photo** and **Batch Photo Walk-through** upload modes.
   - Automatically parses EXIF GPS metadata, displays detected bounding boxes on the uploaded image, renders asset count tables, and highlights loss alerts in red banners.
   - **AI Dynamic Vision Prompt Generator**: Expands natural language landscape descriptions into targeted open-vocabulary detection prompts.
   - **Automated AI Loss Diagnostics**: Generates severity ratings, agronomic root-cause hypotheses, and 24h/7d/30d Embassy SOP remedial action plans with INR liability estimates.
2. **History & Trends Tab**:
   - Allows users to filter by Zone and Asset Category to view historical line charts (`st.line_chart`) and inspect raw SQLite database logs.
3. **SME Audit Summary Tab**:
   - Generates current asset count snapshots per zone. Includes one-click **Download Audit (CSV)** and **Download Audit (Excel)** export buttons using pandas and openpyxl.
   - **AI Executive Landscaping Audit Memorandum**: Auto-generates a formal memorandum summarizing tech park flora, compliance status, and loss risks.
4. **Model Evaluation & Benchmark Tab**:
   - Displays live quantitative metrics (MAE, MAPE, Count Bias), category error summaries, full ground-truth dataset breakdowns, and viva talking points.
5. **AI Landscape Copilot Tab**:
   - Conversational assistant powered by the Hybrid LLM Advisor, answering natural language inquiries regarding census history, zone risk profiles, and Embassy Horticulture SOP specifications.

---

## 5. Verification & Quality Assurance

The codebase includes comprehensive unit test coverage and automation scripts:

- **Automated Test Suites (`tests/`)**: 28 passing unit tests across modules (100% pass rate in 14.48s):
  - `test_asset_register.py`: 12 tests covering SQLite operations, data validation rules, loss alert threshold boundary conditions, and audit export routines.
  - `test_evaluate.py`: 2 tests verifying benchmark fixture integrity and synthetic evaluation dataset generation.
  - `test_llm_advisor.py`: 14 tests covering offline agronomic loss diagnosis, severity ranking, SOP remedial action synthesis, executive memo generation, dynamic prompt expansion, visual health diagnostics, Copilot reasoning, and cloud fallback.
- **Build Automation (`Makefile`)**:
  - `make setup`: Installs pinned dependencies (`requirements.txt`).
  - `make sample_data`: Generates synthetic test fixtures with embedded EXIF GPS tags.
  - `make test`: Executes unit test suites.
  - `make batch`: Runs CLI batch detection on a folder of images.
  - `make evaluate`: Runs model accuracy evaluation benchmark.
  - `make run`: Launches Streamlit dashboard server.

---

## 6. Viva Defense Talking Points (Computer Vision & LLM Architecture)

### Q1: "Why integrate an LLM layer with OWL-ViT instead of relying purely on computer vision?"
> **Defense**: *"OWL-ViT excels at spatial grounding and object detection—telling us what is present and counting assets. However, raw counts alone cannot evaluate physiological stress, explain why an asset count suddenly dropped, or prescribe operational remedies. The Hybrid LLM Intelligence Suite bridges the gap between perception and operational decision-making: it interprets rolling baseline anomalies, determines probable agronomic root causes (such as STP water salinity spikes or root rot), generates an actionable 24h/7d/30d SOP remedial plan, and drafts executive audit memorandums for facility leadership."*

### Q2: "How does your LLM architecture ensure reliability in field conditions without constant internet or API keys?"
> **Defense**: *"We engineered a Hybrid Architecture featuring a zero-dependency, deterministic Agronomic Expert Engine pre-grounded in Embassy Horticulture SOPs and Bengaluru climate flora. If an API key or internet connection is unavailable during site walk-throughs or audits, the system operates completely offline without degradation or exceptions, while still supporting live cloud models (Google Gemini / OpenAI) when connected."*

### Q3: "How does Dynamic Prompt Expansion improve OWL-ViT detection?"
> **Defense**: *"OWL-ViT is an open-vocabulary detector whose accuracy depends directly on prompt text quality. Generic prompts like 'a tree' or 'a shrub' can miss distinct landscape features. The LLM expands high-level supervisor descriptions (e.g. 'tropical poolside with bougainvillea and royal palms') into targeted visual phrases ('a tall royal palm tree', 'a flowering bougainvillea shrub') while maintaining an automatic mapping back to standardized SOP categories in the SQLite register."*

---

## 7. Limitations & Future Development Roadmap

### Known Limitations
1. **Single-Perspective Occlusion**: Trees obscured directly behind larger canopies cannot be detected from a single 2D angle.
2. **Species-Level Specificity**: Generic text prompts (`"a tree"`) categorize broad asset classes but do not distinguish exact species (e.g., Oak vs Birch) without species-specific fine-tuning.

### Future Development Roadmap
- **MobileNetV2 Health Model Integration**: Connect the plant health classifier stub (`classify_plant_health`) to Nikita's trained MobileNetV2 foliage disease model to report asset condition alongside counts.
- **WhatsApp Webhook Ingestion**: Deploy automated webhooks to process site photos posted directly to Embassy operational WhatsApp groups without manual file uploads.
- **Fine-Tuning Vision Backbones**: Fine-tune OWL-ViT's vision transformer backbone on local Indian flora datasets to improve precision under dense foliage.
