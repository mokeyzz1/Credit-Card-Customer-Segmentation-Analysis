# Credit Card Customer Segmentation Analysis

> **Business Impact**: Identified $877K annual losses and developed segmentation strategy with $500K+ recovery potential

## Overview

This project analyzes credit card customer data to uncover profitability insights and develop targeted business strategies. Through machine learning clustering and statistical analysis, I discovered that traditional "high-value" customers are actually the least profitable due to rewards program economics.

## Key Findings

- **Crisis Discovery**: Portfolio losing $877K annually across 30,000 customers
- **Root Cause**: Rewards costs ($128) nearly equal total revenue ($130) per customer  
- **Counter-Intuitive Insight**: High-spending customers generate largest losses (-$83.69 vs -$10.82 for best segment)
- **Solution**: Target "Survivor" segment (older, lower-income customers) for optimal profitability

## Technical Approach

### Data Analysis
- **ETL Pipeline**: Data cleaning, SQL analysis, and database creation
- **Exploratory Analysis**: Statistical analysis of customer demographics and spending behavior
- **Customer Segmentation**: K-means clustering with 4 distinct customer groups

### Statistical Validation
- **ANOVA Testing**: F-statistic validation confirms segment differences (p < 0.001)
- **Stability Analysis**: Clustering consistency across multiple random seeds
- **PCA Visualization**: 2D dimensionality reduction for segment interpretation

## Technologies Used

**Languages & Libraries**: Python, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly  
**Database**: SQLite for SQL analysis  
**Machine Learning**: K-means clustering, PCA, statistical validation  
**Visualization**: Radar charts, scatter plots, statistical charts

## Project Structure

```
├── notebooks/
│   ├── etl_and_sql_analysis.ipynb                # ETL + SQL exploration + crisis discovery
│   └── eda_credit_card_loss_analysis.ipynb       # Customer segmentation + recommendations
├── scripts/
│   └── data_generation.py                        # Data generation pipeline
└── data/
    ├── credit_card_data.csv                      # Source dataset (30K customers)
    └── cleaned_credit_data.csv                   # Processed dataset
```

## Business Recommendations

### Target Segment: "The Survivors" (30.6% of customers)
- **Profile**: Age 57, Income $25K, Spending $808/month
- **Performance**: -$10.82 average loss (best profitability)
- **Strategy**: Focus retention and acquisition efforts

### Restructure Segment: "Premium Disaster" (11.3% of customers)  
- **Profile**: Age 42, Income $141K, Spending $20K/month
- **Performance**: -$83.69 average loss (worst profitability)
- **Strategy**: Reduce rewards or increase fees

## Results & Impact

- **Identified** specific customer segments driving losses
- **Quantified** profitability by demographic and behavioral characteristics  
- **Developed** actionable turnaround strategy with $500K+ potential savings
- **Challenged** traditional banking assumptions about customer value

## Getting Started

1. **View Analysis**: Start with `etl_and_sql_analysis.ipynb` for the ETL process and crisis discovery
2. **Explore Segmentation**: Review `eda_credit_card_loss_analysis.ipynb` for detailed customer segmentation and business recommendations
3. **Run Code**: All notebooks are self-contained with clear documentation

---

