"""
Customer data query API for the Telco churn dataset.

Run:
    uvicorn api:app --reload --port 8001

Docs:
    http://localhost:8001/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

CSV_FILE_PATH = Path("data/telco_churn.csv")

# Global in-memory state (loaded once at startup)
state: dict = {}

YesNo = Literal["Yes", "No"]

SORTABLE_COLUMNS = Literal[
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"
]


def load_data() -> pd.DataFrame:
    """Load and lightly clean the CSV so it is safe to filter and serialize."""
    if not CSV_FILE_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_FILE_PATH)

    # TotalCharges contains blank strings for new customers in the raw Telco data
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Raw dataset stores SeniorCitizen as 0/1; expose it as Yes/No like the other columns
    if "SeniorCitizen" in df.columns and pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["df"] = load_data()
    yield
    state.clear()


app = FastAPI(
    title="Telco Customer Data API",
    description="Query and filter customer records from the Telco churn dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_df() -> pd.DataFrame:
    df = state.get("df")
    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No customer data loaded. Expected CSV at '{CSV_FILE_PATH}'.",
        )
    return df


@app.get("/")
async def root():
    df = state.get("df")
    return {"status": "ok", "rows_loaded": 0 if df is None else len(df)}


@app.get("/customers")
async def list_customers(
    # --- text filters ---
    customer_id: Annotated[
        str | None, Query(description="Partial match on customerID")
    ] = None,
    # --- categorical filters (exact match) ---
    gender: Annotated[Literal["Male", "Female"] | None, Query()] = None,
    senior_citizen: Annotated[YesNo | None, Query()] = None,
    partner: Annotated[YesNo | None, Query()] = None,
    dependents: Annotated[YesNo | None, Query()] = None,
    contract: Annotated[
        Literal["Month-to-month", "One year", "Two year"] | None, Query()
    ] = None,
    internet_service: Annotated[
        Literal["DSL", "Fiber optic", "No"] | None, Query()
    ] = None,
    payment_method: Annotated[
        Literal[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        | None,
        Query(),
    ] = None,
    churn: Annotated[YesNo | None, Query(description="Filter by churn label")] = None,
    # --- numeric range filters ---
    min_tenure: Annotated[int | None, Query(ge=0)] = None,
    max_tenure: Annotated[int | None, Query(ge=0)] = None,
    min_monthly_charges: Annotated[float | None, Query(ge=0)] = None,
    max_monthly_charges: Annotated[float | None, Query(ge=0)] = None,
    # --- sorting + pagination ---
    sort_by: Annotated[SORTABLE_COLUMNS | None, Query()] = None,
    order: Annotated[Literal["asc", "desc"], Query()] = "asc",
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    df = get_df()
    mask = pd.Series(True, index=df.index)

    if customer_id:
        mask &= df["customerID"].str.contains(customer_id, case=False, na=False, regex=False)

    exact_filters = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "Churn": churn,
    }
    for column, value in exact_filters.items():
        if value is not None and column in df.columns:
            mask &= df[column] == value

    if min_tenure is not None:
        mask &= df["tenure"] >= min_tenure
    if max_tenure is not None:
        mask &= df["tenure"] <= max_tenure
    if min_monthly_charges is not None:
        mask &= df["MonthlyCharges"] >= min_monthly_charges
    if max_monthly_charges is not None:
        mask &= df["MonthlyCharges"] <= max_monthly_charges

    filtered = df[mask]

    if sort_by:
        filtered = filtered.sort_values(sort_by, ascending=(order == "asc"), na_position="last")

    total = len(filtered)
    page = filtered.iloc[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "count": len(page),
        "data": to_records(page),
    }


@app.get("/customers/{customer_id}")
async def get_customer(customer_id: str):
    df = get_df()
    row = df[df["customerID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    return to_records(row)[0]


@app.get("/stats")
async def stats():
    """Quick dataset summary: size, churn rate, and churn rate by contract type."""
    df = get_df()
    result: dict = {"total_customers": len(df)}

    if "Churn" in df.columns:
        churned = df["Churn"] == "Yes"
        result["churned"] = int(churned.sum())
        result["churn_rate"] = round(float(churned.mean()), 4)
        if "Contract" in df.columns:
            result["churn_rate_by_contract"] = (
                churned.groupby(df["Contract"]).mean().round(4).to_dict()
            )

    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in df.columns:
            result[f"avg_{col}"] = round(float(df[col].mean()), 2)

    return result