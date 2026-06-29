# NEMT Trip Cancellation Predictor
**Predicting trip loss before it happens — built on real operational data**

---

## The problem
In Non-Emergency Medical Transportation, ~28% of scheduled trips
are lost to cancellations and no-shows. For a fleet of 70+ drivers
handling 15,000+ trips/month, that's tens of thousands in missed
revenue — every month.

## What this project does
Uses machine learning to predict whether a scheduled trip will be
completed or lost, based on:
- Broker (ModivCare / CTS / SafeRide / MediDrive)
- Driver assignment
- Pickup time and day
- Trip price and mileage

## Results
| Model              | Lost trips found | Accuracy |
|--------------------|-----------------|----------|
| Baseline (dummy)   | 0 / 864 (0%)    | 72%      |
| Decision Tree v1   | 382 / 864 (44%) | 82%      |
| Decision Tree v2   | 760 / 864 (88%) | 95%      |

Key finding: **driver assignment** explains 63% of cancellation
variance — more than broker, time of day, or trip price combined.

## Business impact
Identifying high-risk trips before dispatch enables proactive
reassignment → estimated $34,000/month in recoverable revenue.

## Stack
Python · pandas · scikit-learn · openpyxl

## Data
Anonymized RouteGenie export — June 2026, 15,364 trips across
5 brokers and 70+ drivers.

## Files
nemt_analysis.py        — main pipeline
driver_diagnosis.xlsx   — per-driver cancellation breakdown
README.md               — this file
