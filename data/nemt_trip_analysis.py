"""
NEMT Trip Cancellation Analysis
================================
Predicting trip loss before it happens — built on real operational data.

Author: Timur Trubitsin
GitHub: github.com/ttrubitsin/nemt-trip-analysis
Data:   RouteGenie export, June 2026, 15,364 trips, 5 brokers, 70+ drivers
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_and_merge(billing_path: str, geo_path: str) -> pd.DataFrame:
    """
    Load billing report and geo report, merge on Pick Up Time + Order Price.
    Both files are RouteGenie exports from the same period.
    """
    billing = pd.read_excel(billing_path, sheet_name="report", header=1)
    geo     = pd.read_excel(geo_path,     sheet_name="report", header=1)

    billing["Pick Up Time"] = pd.to_datetime(billing["Pick Up Time"])
    geo["Pick Up Time"]     = pd.to_datetime(geo["Pick Up Time"])

    # Join key: timestamp + price uniquely identifies a trip across both files
    billing["_key"] = billing["Pick Up Time"].astype(str) + "_" + billing["Order Price"].astype(str)
    geo["_key"]     = geo["Pick Up Time"].astype(str)     + "_" + geo["Order Price"].astype(str)

    geo_slim = geo[["_key", "Order Mileage", "Pick Up Address",
                    "Order Drop Off Address"]].drop_duplicates("_key")

    df = billing.merge(geo_slim, on="_key", how="left").drop(columns=["_key"])
    print(f"Loaded {len(df):,} trips | {df['Pick Up Time'].min().date()} → {df['Pick Up Time'].max().date()}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variable and encoded features."""
    df = df.copy()

    # Target: 1 = trip lost (Canceled or No Show), 0 = Completed
    df["Lost"]         = df["Order Status"].isin(["Canceled", "No show"]).astype(int)
    df["Canceled_flag"]= (df["Order Status"] == "Canceled").astype(int)
    df["NoShow_flag"]  = (df["Order Status"] == "No show").astype(int)

    # Time features
    df["Hour"]      = df["Pick Up Time"].dt.hour
    df["DayOfWeek"] = df["Pick Up Time"].dt.dayofweek   # 0 = Monday

    # Categorical encodings (label encoding — sufficient for tree models)
    df["Broker_code"]   = df["Payer Name"].astype("category").cat.codes
    df["Driver_code"]   = df["Full name"].astype("category").cat.codes
    df["Passenger_code"]= df["Passenger's Full Name"].astype("category").cat.codes

    # City extracted from address string
    df["PickUp_city"]  = df["Pick Up Address"].str.extract(r",\s*([^,]+),\s*[A-Z]{2}\s*\d")
    df["DropOff_city"] = df["Order Drop Off Address"].str.extract(r",\s*([^,]+),\s*[A-Z]{2}\s*\d")
    df["PickUp_city_code"]  = df["PickUp_city"].astype("category").cat.codes
    df["DropOff_city_code"] = df["DropOff_city"].astype("category").cat.codes

    return df


# ─────────────────────────────────────────────
# 3. BASELINE MODEL
# ─────────────────────────────────────────────

def run_baseline(X_train, X_test, y_train, y_test) -> None:
    """
    Baseline: always predict the majority class (Completed).
    This sets the floor — any real model must beat this.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    print("\n" + "="*50)
    print("BASELINE MODEL — always predicts 'Completed'")
    print("="*50)
    print(f"Accuracy : {100*accuracy_score(y_test, y_pred):.1f}%")
    print(classification_report(y_test, y_pred,
                                 target_names=["Completed", "Lost"]))
    lost_found = ((y_pred == 1) & (y_test == 1)).sum()
    print(f"Lost trips found: {lost_found} / {y_test.sum()} → {100*lost_found/y_test.sum():.0f}%")


# ─────────────────────────────────────────────
# 4. DECISION TREE MODEL
# ─────────────────────────────────────────────

def run_decision_tree(X_train, X_test, y_train, y_test,
                      features: list, version: str = "v1",
                      max_depth: int = 5) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier and print results.
    class_weight='balanced' compensates for class imbalance (~28% lost).
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"DECISION TREE {version.upper()} — {len(features)} features")
    print("="*50)
    print(f"Accuracy : {100*accuracy_score(y_test, y_pred):.1f}%")
    print(classification_report(y_test, y_pred,
                                 target_names=["Completed", "Lost"]))

    lost_found = ((y_pred == 1) & (y_test == 1)).sum()
    print(f"Lost trips found: {lost_found} / {y_test.sum()} → {100*lost_found/y_test.sum():.0f}%")

    print("\nFeature importance:")
    for feat, imp in sorted(zip(features, model.feature_importances_),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"  {feat:<22}: {imp:.3f} {bar}")

    return model


# ─────────────────────────────────────────────
# 5. DRIVER DIAGNOSIS
# ─────────────────────────────────────────────

def driver_diagnosis(df: pd.DataFrame, min_trips: int = 10) -> pd.DataFrame:
    """
    Classify each driver by cancellation pattern:
    - 'Passenger issue'  → No Show > 60% of losses  (not driver's fault)
    - 'Driver issue'     → Canceled > 70% of losses (talk to driver)
    - 'Mixed'            → investigate case by case
    - 'Normal'           → loss rate < 10%
    """
    stats = df.groupby("Full name").agg(
        Total   =("Lost", "count"),
        Lost    =("Lost", "sum"),
        Canceled=("Canceled_flag", "sum"),
        NoShow  =("NoShow_flag",   "sum"),
        Earned  =("Final Price",   "sum"),
    ).reset_index()

    stats["pct_lost"]    = 100 * stats["Lost"]     / stats["Total"]
    stats["pct_canceled"]= 100 * stats["Canceled"] / stats["Total"]
    stats["pct_noshow"]  = 100 * stats["NoShow"]   / stats["Total"]
    stats["ns_share"]    = 100 * stats["NoShow"]   / stats["Lost"].replace(0, 1)

    stats = stats[stats["Total"] >= min_trips].copy()

    def classify(row):
        if row["pct_lost"] < 10:
            return "✅ Normal"
        if row["ns_share"] >= 60:
            return "📋 Passenger issue (No Show)"
        elif row["ns_share"] <= 30:
            return "⚠️  Talk to driver (Canceled)"
        return "🔍 Mixed — investigate"

    stats["Diagnosis"] = stats.apply(classify, axis=1)
    return stats.sort_values(["Diagnosis", "pct_lost"],
                              ascending=[True, False]).reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. BROKER SUMMARY
# ─────────────────────────────────────────────

def broker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and loss breakdown by broker."""
    return df.groupby("Payer Name").agg(
        Total   =("Lost", "count"),
        Completed=("Lost", lambda x: (x == 0).sum()),
        Lost    =("Lost", "sum"),
        Earned  =("Final Price", "sum"),
        Lost_rev=("Order Price", lambda x: x[df.loc[x.index, "Lost"] == 1].sum()),
    ).assign(
        pct_lost=lambda d: (100 * d["Lost"] / d["Total"]).round(1)
    ).sort_values("pct_lost", ascending=False)


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Paths (update to your actual file locations) ──────────────────────
    BILLING_PATH = "data/billing_june2026.xlsx"
    GEO_PATH     = "data/geo_june2026.xlsx"
    # ──────────────────────────────────────────────────────────────────────

    # Load & prepare
    df = load_and_merge(BILLING_PATH, GEO_PATH)
    df = build_features(df)

    # ── Quick stats ──────────────────────────────────────────────────────
    total = len(df)
    lost  = df["Lost"].sum()
    print(f"\nOverall loss rate: {lost:,} / {total:,} = {100*lost/total:.1f}%")

    print("\nBroker summary:")
    print(broker_summary(df).to_string())

    # ── Model v1: basic features ─────────────────────────────────────────
    features_v1 = ["Hour", "DayOfWeek", "Broker_code", "Order Price"]
    df_v1 = df[features_v1 + ["Lost"]].dropna()
    X1, y1 = df_v1[features_v1], df_v1["Lost"]
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1)

    run_baseline(X1_tr, X1_te, y1_tr, y1_te)
    run_decision_tree(X1_tr, X1_te, y1_tr, y1_te, features_v1, version="v1")

    # ── Model v2: full features (driver, passenger, city, mileage) ───────
    features_v2 = ["Hour", "DayOfWeek", "Broker_code", "Order Price",
                   "Order Mileage", "Driver_code", "Passenger_code",
                   "PickUp_city_code", "DropOff_city_code"]
    df_v2 = df[features_v2 + ["Lost"]].dropna()
    X2, y2 = df_v2[features_v2], df_v2["Lost"]
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2)

    run_decision_tree(X2_tr, X2_te, y2_tr, y2_te, features_v2, version="v2",
                      max_depth=5)

    # ── Driver diagnosis ─────────────────────────────────────────────────
    print("\n" + "="*50)
    print("DRIVER DIAGNOSIS (10+ trips)")
    print("="*50)
    diag = driver_diagnosis(df)
    print(diag[["Full name", "Total", "pct_lost",
                "pct_canceled", "pct_noshow", "Diagnosis"]].to_string(index=False))

    print("\nDone. Results above — use Excel output for full formatted report.")


if __name__ == "__main__":
    main()
