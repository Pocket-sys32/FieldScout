# Cache Creek Game Camera Project

Automated species identification for Bushnell trail-cam video files.
Detects 15 wildlife targets plus humans, counts simultaneous animals, and writes results directly to your Google Sheet.

---

## Quick Start (Volunteers)

1. **Install Python 3.10+** from [python.org](https://python.org) — check "Add Python to PATH" during install.
2. **Double-click `run.bat`** — it installs all dependencies automatically on first launch (takes ~5 min and needs internet; subsequent launches are instant).
   Each normal launch also checks the configured Git upstream, installs a safe fast-forward update when the checkout has no local edits, and refreshes changed dependencies inside `.venv`.
3. **First run downloads the AI models** (~500 MB total, cached locally — only once).
4. In the app, click **⚙ Settings** and fill in:
   - *Google Sheet ID* — the long string in your Sheet's URL between `/d/` and `/edit`
   - *Service account JSON path* — path to the key file (see Google Sheets Setup below)
5. **Browse or drag** a folder of `.mov` files onto the drop zone.
6. Click **▶ Process Videos** — results appear in the log and are written to the Sheet in real time.

Settings are stored in the local `config.json`. That file and the service-account key are excluded from Git, so each computer keeps its own credentials and Sheet configuration during updates.

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
| California Ground Squirrel | *Otospermophilus beecheyi* |
| Fox Squirrel | *Sciurus niger* |
| Bird | *Aves spp.* |
| California Quail | *Callipepla californica* |
| Golden-crowned Sparrow | *Zonotrichia atricapilla* |
| North American River Otter | *Lontra canadensis* |
| Human | *Homo sapiens* |

---

## SpeciesNet Setup (one-time, free)

SpeciesNet is Google's camera-trap AI, trained on millions of night-IR and day images from Wildlife Insights. It handles Bushnell IR footage better than the general-purpose CLIP fallback. The Python dependency is installed with FieldScout, and its model downloads automatically on first use. If the model cannot load or download, FieldScout logs the reason and continues with CLIP.

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
Date    Time    Species Captured    Scientific Name    Numbers of same species present    picture number    Comment    Confidence Int.    Review
```

The app will append one row below the header per processed video.

---

## Google Sheet Columns Explained

| Column | Description |
|---|---|
| **Date** | Recording date (MM/DD/YYYY) extracted from video metadata |
| **Time** | Recording time (`hh:mm:ss AM/PM`) preserved from the camera clock |
| **Species Captured** | Detected species common name |
| **Scientific Name** | Binomial nomenclature |
| **Numbers of same species present** | Conservative median number of matching animals seen per sampled frame |
| **picture number** | Source video file name |
| **Comment** | Auto-notes: "Night IR", possible second species, etc. |
| **Confidence Int.** | AI confidence score (0.000–1.000) |
| **Review** | `TRUE` when confidence < 65% or two species are nearly tied |

---

## CLI Batch Mode (advanced)

Run without a GUI — useful for scheduled overnight processing:

```bat
python main.py --batch C:\path\to\video_folder
```

Results still go to the Sheet and `detections.csv`.

Use `python main.py --no-update` to launch without the Git update check.

---

## Correcting Results

Sheet entries can be edited manually. The dropdown for **Species Captured** should include the exact output labels above, plus **No Animal Detected** and **Unknown Animal**. A Sheet correction changes that record only; the current model does not read corrections back from Google Sheets or retrain itself.

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
│   ├── species.py         Species registry + CLIP prompts
│   ├── video.py           .mov frame extraction + Bushnell timestamp parsing
│   ├── detector.py        MegaDetector v6 wrapper
│   ├── classifier.py      CLIP zero-shot + ONNX Phase-2 classifier
│   ├── aggregate.py       Per-video result aggregation
│   ├── pipeline.py        End-to-end processing orchestrator
│   ├── sheets.py          Google Sheets + CSV writer
│   └── gui.py             CustomTkinter desktop GUI
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

**Wrong species identified** — Correct the Sheet row. This fixes the record but does not retrain SpeciesNet or CLIP.

**Automatic update skipped** — FieldScout protects local work. Commit or stash tracked changes and make sure the current branch has a configured Git upstream. The app continues using the installed version when offline.
