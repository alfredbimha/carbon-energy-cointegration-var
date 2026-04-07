# Carbon Prices and Energy Stock Cointegration (VAR)

## Research Question
Are carbon prices, clean energy, and fossil fuel stocks cointegrated?

## Methodology
**Language:** Python  
**Methods:** Johansen cointegration, VAR, Granger causality, IRF

## Data
Yahoo Finance — KRBN (carbon ETF), ICLN, XLE (2020–2025)

## Key Findings
Evidence of cointegration; clean energy Granger-causes fossil fuel returns; impulse responses show cross-market transmission.

## How to Run
```bash
pip install -r requirements.txt
python code/project4_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
