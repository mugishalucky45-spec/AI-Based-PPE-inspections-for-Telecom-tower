
# Telecom Tower PPE Inspection Monitoring System

An AI-powered project that uses YOLOv8 and Streamlit to help monitor Personal Protective Equipment (PPE) at telecommunications tower sites.

## Features

- Site information and worker identification
- Safety hazard reporting
- PPE detection using YOLOv8
- Confidence scores for detected PPE
- Worker inspection status lookup

## PPE Classes

- Helmet
- Safety gloves
- Safety vest
- Safety shoes
- Full-body safety harness

## Technologies Used

- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- Pandas
- NumPy

## Installation

1. Install Python.
2. Open a terminal in the project folder.
3. Install the required packages:

   `pip install -r requirements.txt`

4. Make sure your trained model (`best.pt`) is available at the model path configured in `app.py`.
5. Start the application:

   `python -m streamlit run app.py`

## Disclaimer

This is an educational prototype. PPE detections may be incorrect or incomplete and must not replace qualified safety inspections.
