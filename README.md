# FieldScout//

**FieldScout** is a computer vision project designed to help conservancies process trail camera footage into usable data.

## Features:
- **AI Detection:** Automatically identify animals using a custom-trained YOLO model
- **Time Stamping:** Uses OCR to scrape timestamps from the video
- **Google Sheets Integration:** Hooks into any google sheet to input data
- **Two Modes:** Uses Batch-Processing for folders and individual processes.

## Stack:
- **Python 3.11**
- **CV:** Ultralytics Yolov8, OpenCV, EasyOCR
- **UI:** Steamlit
- **Data:** Google Sheets API, Pandas
- **Model trained by:** Roboflow

To use this project:
- Clone the repo
- Set up .venv
- Install Dependencies
- Run "streamlit run app/ui.py" in terminal


![FieldScout Demo](assets/demo.gif) 
