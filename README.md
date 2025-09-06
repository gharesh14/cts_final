🏥 Prior Authorization Automation System
📌 Overview

This project automates prior authorization (PA) workflows by combining rule-based checks, machine learning models, and LLM-powered decision support. It helps providers reduce administrative burden, minimize delays, and improve care outcomes.

📂 Repository Structure
├── static/                     # Frontend static assets (CSS, JS, images)
├── templates/                  # HTML frontend templates
├── app.py                      # Main Flask backend
├── appeal_risk_model.pkl_first # ML appeal risk model (v1)
├── appeal_risk_model.pkl_second
├── appeal_risk_model.pkl_third
├── xgb_appeal_model.pkl        # XGBoost appeal model
├── rules_1000.json             # Rule engine JSON (payer policies)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Runtime environment (for deployment)
├── README.md                   # Project documentation

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Initialize Database
python
>>> from app import db
>>> db.create_all()

4️⃣ Run App
python app.py


App runs at: http://127.0.0.1:5000

🚀 Key Features

Frontend (provider portal: HTML/CSS/JS)

Rule Engine (reads rules_1000.json, validates requests against payer policies)

Appeal Risk Models (multiple trained models incl. XGBoost)

Eligibility & Coverage Checks (FHIR-compliant endpoint)

Audit Trail (all requests, approvals, denials, and appeals logged)


Appeal Risk Models: XGBoost Classifer.

Training Data: Synthetic datasets (specialty meds, imaging, procedures, DME).

Target: Appeal success probability.


✅ Future Enhancements

Cloud deployment with PostgreSQL & Docker.

Expand payer coverage (UHC, Aetna, Cigna).

Improve ML accuracy >90% with real-world data.

Full EMR/EHR integration with HL7 FHIR.
