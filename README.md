
# 🏦 SME Alternative Credit Scoring & XAI Engine

A production-grade algorithmic underwriting system designed for Fintechs and NBFCs to evaluate Small and Medium Enterprise (SME) loan applications. This system bypasses traditional credit history requirements by combining real-world financial data with alternative risk metrics, utilizing Explainable AI (XAI) to maintain strict regulatory compliance.

## 📌 Project Overview
Traditional banking infrastructure often automatically rejects early-stage B2B enterprises that lack historical sales data. This engine solves that problem through a **Hybrid ML Approach**:
1. It grounds its baseline risk calculations in the highly respected **U.S. Small Business Administration (SBA) dataset**, analyzing core metrics like loan term, job creation, and disbursement amounts.
2. It mathematically injects **Alternative Credit Metrics** (GST Filing Consistency, Local Market Growth, Social Media Sentiment) to dynamically adjust the risk profile.
3. It utilizes **SHAP (SHapley Additive exPlanations)** to break open the XGBoost "black box," generating a visual waterfall plot that explicitly proves to a human underwriter *why* a loan was approved or rejected, satisfying RBI/regulatory requirements for algorithmic transparency.

## ⚙️ The Architecture

* **The Data Pipeline:** Fetches, cleans, and engineers a hybrid dataset combining real SBA default histories with synthetic alternative data parameters.
* **The Brain (XGBoost):** A Gradient Boosting Classifier trained to predict the probability of loan default based on non-linear feature interactions.
* **The XAI Engine (SHAP):** Calculates the exact point contribution of every single variable to bridge the gap between the baseline risk and the final algorithmic decision.
* **The Front-End (Streamlit):** A reactive underwriter-facing dashboard for real-time application simulation and visual risk analysis.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-learn
* **Explainable AI:** SHAP
* **Data Manipulation:** Pandas, NumPy
* **UI & Deployment:** Streamlit, Matplotlib

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/nikamsudarshan/SME-Credit-Scoring.git
cd SME-Credit-Scoring

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```

3. Launch the Underwriter Dashboard:
```bash
streamlit run src/app.py

```

*Note: Upon the first launch, the engine will dynamically download the dataset, inject the alternative metrics, and train the XGBoost model in the background before rendering the UI.*

