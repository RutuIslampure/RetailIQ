# RetailIQ — Customer & Sales Intelligence Platform

An end-to-end retail analytics platform built on 400K+ real transactions 
from the UCI Online Retail II dataset.

## Live Dashboard
🔗 [View Live App](https://rutuislampure-retailiq.streamlit.app)

## Project Overview
- **Dataset:** UCI Online Retail II (400K+ transactions, Dec 2009 – Nov 2011)
- **Domain:** Retail Analytics / Customer Intelligence

## Features
- 📊 **Business Overview** — KPIs, revenue trends, top products and countries
- 👥 **Customer Analysis** — RFM segmentation with interactive scatter plots
- ⚠️ **Churn Prediction** — Real-time ML prediction with gauge chart
- 📈 **Sales Forecast** — 3-month revenue forecasting with seasonal features

## Tech Stack
- **Language:** Python
- **Analytics:** Pandas, NumPy, Scikit-learn
- **Visualisation:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Database:** PostgreSQL
- **ML Models:** Logistic Regression, Random Forest, Gradient Boosting
- **Deployment:** Streamlit Cloud, Git/GitHub

## Run with Docker
The easiest way to run this project locally — no need to manually install Python packages.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

```bash
git clone https://github.com/Rutulslampure/RetailIQ.git
cd RetailIQ
docker compose up --build
```

Then open http://localhost:8502 in your browser.

See [DOCKER.md](./DOCKER.md) for more details.

## ML Results
| Model | AUC Score |
|---|---|
| Logistic Regression | 0.740 |
| Gradient Boosting | 0.731 |
| Random Forest | 0.643 |

- **Churn Recall:** 73% (catches 73% of at-risk customers)
- **Forecast R²:** 0.459 with seasonal features

## Project Structure
RetailIQ/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── retail_clean.csv        # Cleaned transaction data
├── rfm_segments.csv        # RFM customer segments
├── monthly_revenue.csv     # Monthly revenue with seasonal features
├── churn_model.pkl         # Trained churn prediction model
├── scaler.pkl              # Feature scaler
└── forecast_model.pkl      # Revenue forecasting model

## Author
**Rutu Shivanand Islampure**  
B.E. CSE (AI & ML) — AMC Engineering College, Bangalore  
[LinkedIn](https://www.linkedin.com/in/rutu-islampure-86041b294) | 
[GitHub](https://github.com/RutuIslampure)
