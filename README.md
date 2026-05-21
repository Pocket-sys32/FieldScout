# Cache Creek Game Camera Project

Automated species identification for Bushnell trail-cam `.mov` files.
Detects the 13 target species, counts simultaneous animals, and writes results directly to your Google Sheet.

---

## Quick Start (Volunteers)

1. **Install Python 3.10+** from [python.org](https://python.org) — check "Add Python to PATH" during install.
2. **Double-click `run.bat`** — it installs all dependencies automatically on first launch (takes ~5 min and needs internet; subsequent launches are instant).
3. **First run downloads the AI models** (~500 MB total, cached locally — only once).
4. In the app, click **⚙ Settings** and fill in:
   - *Google Sheet ID* — the long string in your Sheet's URL between `/d/` and `/edit`
   - *Service account JSON path* — path to the key file (see Google Sheets Setup below)
5. **Browse or drag** a folder of `.mov` files onto the drop zone.
6. Click **▶ Process Videos** — results appear in the log and are written to the Sheet in real time.

---

## Target Species

| Common Name | Scientific Name |
|---|---|
| Beaver | *Castor canadensis* |
| Bobcat | *Lynx rufus* |
| Coyote | *Canis latrans* |
| Striped Skunk | *Mephitis mephitis* |
| Virginia Opossum | *Didelphis virginiana* |
| Columbian Black-tailed Deer | *Odocoileus hemionus columbianus* |
| Gray Fox | *Urocyon cinereoargenteus* |
| Raccoon | *Procyon lotor* |
| Desert Cottontail | *Sylvilagus audubonii* |
| Squirrel | *Sciuridae spp.* |
| California Quail | *Callipepla californica* |
| Golden-crowned Sparrow | *Zonotrichia atricapilla* |
| North American River Otter | *Lontra canadensis* |

---

## SpeciesNet Setup (one-time, free — strongly recommended for IR footage)

SpeciesNet is Google's camera-trap AI, trained on millions of night-IR and day images from Wildlife Insights. It handles Bushnell IR footage much better than CLIP.

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to **Account → Settings → API → Create New Token**
3. A file called `kaggle.json` downloads — save it to:
   ```
   C:\Users\<your username>\.kaggle\kaggle.json
   ```
   (Create the `.kaggle` folder if it doesn't exist)
4. Run the app — SpeciesNet downloads automatically (~500 MB, once only)

The app falls back to CLIP automatically if `kaggle.json` is missing, so this step is optional but recommended.

---

## Google Sheet Setup (one-time)

### Step 1 — Create a Google Cloud service account

1. Go to [console.cloud.google.com](https://console.cloud.google.com).
2. Create a new project (e.g. *Cache Creek Game Camera*).
3. In the sidebar → **APIs & Services** → **Enable APIs** → search for **Google Sheets API** → Enable.
4. In the sidebar → **IAM & Admin** → **Service Accounts** → **Create Service Account**.
   - Name: `fsvcc-detector`
   - Click through (no role needed at this step)
5. Click the new service account → **Keys** tab → **Add Key → Create new key → JSON** → Download.
6. Save the downloaded `.json` file as `service_account.json` in the same folder as `run.bat`.

### Step 2 — Share your Sheet with the service account

1. Open your Google Sheet.
2. Click **Share** (top-right).
3. Paste the service account's email address (visible in the JSON file as `"client_email"`) and give it **Editor** access.

### Step 3 — Set up the Sheet header row

Make sure Row 1 of the target worksheet tab contains exactly these headers (copy-paste):

```
Date    Time    Common Name    Scientific Name    Count    Filename    Comments    Confidence    Needs Review
```

The app will append one row below the header per processed video.

---

## Google Sheet Columns Explained

| Column | Description |
|---|---|
| **Date** | Recording date (MM/DD/YYYY) extracted from video metadata |
| **Time** | Recording time (HH:MM:SS) extracted from video metadata |
| **Common Name** | Detected species common name |
| **Scientific Name** | Binomial nomenclature |
| **Count** | Max number of animals seen simultaneously in any frame |
| **Filename** | Source `.mov` file name |
| **Comments** | Auto-notes: "Night IR", "Multiple species detected", etc. |
| **Confidence** | AI confidence score (0.000–1.000) |
| **Needs Review** | `TRUE` when confidence < 65% or two species are nearly tied |

---

## CLI Batch Mode (advanced)

Run without a GUI — useful for scheduled overnight processing:

```bat
python main.py --batch C:\path\to\video_folder
```

Results still go to the Sheet and `detections.csv`.

---

## Improving Accuracy — Phase 2 Fine-Tuning

**Phase 1** (default) uses CLIP zero-shot — no training needed, ~75–85% accuracy.

**Phase 2** fine-tunes EfficientNet-B0 on your own footage → ~88–95% accuracy.

### Step 1 — Build the review set

Run this after processing a batch (or on your existing archive):

```bat
python scripts\build_review_set.py --videos C:\path\to\movs --output review_crops
```

This saves every detected animal crop as a JPEG organised into per-species folders.

### Step 2 — Verify crops (volunteers)

Open `review_crops\` in File Explorer.  Move mis-labelled images to the correct folder.
You need roughly **200–500 verified images per species** for a good fine-tune.

### Step 3 — Train (GPU workstation / Google Colab)

```bash
# Install training extras
pip install timm onnx onnxruntime

python scripts/train_classifier.py \
    --data   review_crops/ \
    --epochs 30 \
    --output models/fsvcc_classifier.onnx
```

Upload to Google Colab for free GPU if your machine is CPU-only.

### Step 4 — Switch the app to Phase 2

In **⚙ Settings**, set *Custom ONNX classifier path* to `models/fsvcc_classifier.onnx`.
The app switches backends automatically on next launch.

---

## Hardware & Performance

| Metric | Value |
|---|---|
| Hardware required | Modern CPU (no GPU needed) |
| Per-frame detection | ~0.3–1.0 sec |
| Typical 20-sec clip | ~20–40 sec end-to-end |
| 50-clip overnight batch | ~20–35 min |
| Model download (first run) | ~500 MB (cached, not re-downloaded) |

---

## Project Structure

```
FSvCC/
├── fsvcc_detector/        Python package
│   ├── config.py          Settings (persisted to config.json)
│   ├── species.py         13-species registry + CLIP prompts
│   ├── video.py           .mov frame extraction + Bushnell timestamp parsing
│   ├── detector.py        MegaDetector v6 wrapper
│   ├── classifier.py      CLIP zero-shot + ONNX Phase-2 classifier
│   ├── aggregate.py       Per-video result aggregation
│   ├── pipeline.py        End-to-end processing orchestrator
│   ├── sheets.py          Google Sheets + CSV writer
│   └── gui.py             CustomTkinter desktop GUI
├── scripts/
│   ├── build_review_set.py   Export crops for volunteer review
│   └── train_classifier.py   Phase-2 EfficientNet-B0 fine-tuning
├── models/                Downloaded weights cached here
├── main.py                Entry point (GUI or CLI)
├── requirements.txt
├── run.bat                Double-click to launch (Windows)
└── build_exe.bat          Build standalone .exe (advanced)
```

---

## Troubleshooting

**"No .mov files found"** — Check the folder contains `.mov` or `.MOV` files (not `.mp4` or `.avi`).

**"Service account key not found"** — Make sure `service_account.json` is in the same folder as `run.bat`, or set the correct path in Settings.

**"Sheet write failed"** — Verify the service account email has Editor access to the Sheet, and the Sheet ID in Settings is correct.

**Models downloading slowly** — The first run downloads ~500 MB.  Run on a fast internet connection once; subsequent runs are instant.

**Wrong species identified** — Add the clip to the review set and correct the label; this data feeds Phase-2 training and will improve future accuracy.
