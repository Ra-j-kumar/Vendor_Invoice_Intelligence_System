# Vendor Invoice Intelligence System

## Overview
This project is an end-to-end machine learning solution for vendor invoice analysis. It helps finance and procurement teams predict freight costs and detect potentially risky invoices using trained models and an interactive Streamlit dashboard.

---

## Project Workflow

```text
Invoice Data (SQLite Database)
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Trained Models Saved
   ↓
Streamlit Web App for Predictions
```

---

## Project Structure

```text
Vendor_Invoice_Intelligence_System/
│
├── app.py
├── README.md
├── data/
│   └── inventory.db
├── freight_cost_prediction/
│   ├── data_prepossing.py
│   ├── model_evalution.py
│   ├── train.py
│   └── models/
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   ├── train.py
│   └── models/
├── inferences/
│   ├── Predict_freight.py
│   └── Predict_invoice_flag.py
└── notebooks/
```

---

## Features

- Predict freight cost from invoice amount
- Detect invoice risk or suspicious invoices
- Use a simple and interactive Streamlit app
- Train and save machine learning models locally
- Work with SQLite-based invoice data

---

## Technologies Used

- Python
- Streamlit
- pandas
- scikit-learn
- joblib
- SQLite

---

## Installation

1. Clone the repository
2. Navigate to the project folder
3. Install the required libraries

```bash
pip install streamlit pandas scikit-learn joblib
```

---

## Running the App

Start the web application using:

```bash
streamlit run app.py
```

This will open the dashboard where you can:
- predict freight cost
- predict invoice risk flag

---

## Model Training

To retrain the models, run:

```bash
python freight_cost_prediction/train.py
python invoice_flagging/train.py
```

These scripts train the models and save them inside their respective model folders.

---

## Data Source

The project uses the SQLite database file:

```text
data/inventory.db
```

---

## Key Use Case

This system is useful for finance teams who want to:
- reduce manual invoice review
- improve cost forecasting
- detect unusual invoice behavior faster

---

## License
This project is intended for educational and demonstration purposes.
