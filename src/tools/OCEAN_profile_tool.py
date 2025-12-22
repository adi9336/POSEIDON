"""
Ocean Profile Analysis Tool
---------------------------
Agent-callable pipeline for analyzing ocean / ARGO vertical profile CSV files.

Expected CSV columns:
- pres      : pressure (dbar)
- temp      : temperature (°C)
- psal      : salinity (PSU)
- latitude
- longitude
"""

from typing import Dict
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import gsw


# ============================================================
# 1. Quality Control
# ============================================================
def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical QC rules to ocean profile data."""
    df = df.copy()

    df = df[
        (df["pres"] >= 0) &
        (df["temp"].between(-2, 40)) &
        (df["psal"].between(0, 42))
    ]

    df = df.dropna()
    df = df.sort_values("pres")

    return df


# ============================================================
# 2. Vertical Binning
# ============================================================
def bin_vertical_profile(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """Bin data vertically by pressure."""
    df = df.copy()
    df["pres_bin"] = (df["pres"] // step) * step

    return (
        df.groupby("pres_bin")
        .mean(numeric_only=True)
        .reset_index()
    )


# ============================================================
# 3. Smoothing
# ============================================================
def smooth_profile(
    df: pd.DataFrame,
    window: int = 7,
    polyorder: int = 2
) -> pd.DataFrame:
    """Smooth temperature and salinity profiles."""
    df = df.copy()

    df["temp_smooth"] = savgol_filter(df["temp"], window, polyorder)
    df["psal_smooth"] = savgol_filter(df["psal"], window, polyorder)

    return df


# ============================================================
# 4. Derived Physics (TEOS-10)
# ============================================================
def compute_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute density and gradients using TEOS-10."""
    df = df.copy()

    lat = df["latitude"].iloc[0]
    lon = df["longitude"].iloc[0]

    SA = gsw.SA_from_SP(
        df["psal_smooth"],
        df["pres_bin"],
        lon,
        lat
    )

    CT = gsw.CT_from_t(
        SA,
        df["temp_smooth"],
        df["pres_bin"]
    )

    df["density_sigma0"] = gsw.sigma0(SA, CT)

    df["dT_dz"] = np.gradient(df["temp_smooth"], df["pres_bin"])
    df["dS_dz"] = np.gradient(df["psal_smooth"], df["pres_bin"])

    return df


# ============================================================
# 5. Insight Extraction
# ============================================================
def extract_insights(df: pd.DataFrame) -> Dict:
    """Extract physically meaningful insights."""
    thermocline_depth = float(
        df.loc[df["dT_dz"].abs().idxmax(), "pres_bin"]
    )

    stable_density = bool(
        np.all(np.diff(df["density_sigma0"]) > 0)
    )

    return {
        "surface_temp": float(df["temp_smooth"].iloc[0]),
        "bottom_temp": float(df["temp_smooth"].iloc[-1]),
        "surface_salinity": float(df["psal_smooth"].iloc[0]),
        "bottom_salinity": float(df["psal_smooth"].iloc[-1]),
        "thermocline_depth_dbar": thermocline_depth,
        "stable_density_column": stable_density
    }


# ============================================================
# 6. MAIN PIPELINE FUNCTION (TOOL CORE)
# ============================================================
def analyze_ocean_profile(
    csv_path: str,
    bin_step: int = 10,
    smooth_window: int = 7
) -> Dict:
    """
    Analyze an ocean CSV vertical profile and return
    structured physics-based insights.
    """

    df = pd.read_csv(csv_path)

    df = quality_control(df)
    df = bin_vertical_profile(df, bin_step)
    df = smooth_profile(df, smooth_window)
    df = compute_physics(df)

    summary = extract_insights(df)

    return {
        "summary": summary,
        "profiles": {
            "pressure": df["pres_bin"].tolist(),
            "temperature": df["temp_smooth"].tolist(),
            "salinity": df["psal_smooth"].tolist(),
            "density": df["density_sigma0"].tolist(),
        }
    }


# ============================================================
# 7. LANGCHAIN TOOL WRAPPER
# ============================================================
from langchain.tools import tool

@tool
def ocean_profile_analysis_tool(
    csv_path: str,
    bin_step: int = 10,
    smooth_window: int = 7
) -> Dict:
    """
    Analyze an ocean / ARGO CSV profile and return
    thermocline depth, stratification, and stability insights.
    """
    return analyze_ocean_profile(csv_path, bin_step, smooth_window)


# ============================================================
# 8. MAIN (TEST + AGENT EXECUTION)
# ============================================================
if __name__ == "__main__":

    # ---------- BASIC PIPELINE TEST ----------
    csv_path = r"C:/Users/ADITYA GUPTA/POSEIDON/data/argo_data_20251221_173238.csv"

    print("\nRunning direct pipeline test...\n")
    result = analyze_ocean_profile(csv_path)

    for k, v in result["summary"].items():
        print(f"{k}: {v}")

    # ---------- AGENT TEST ----------
    print("\nRunning agent test...\n")

    from langchain_openai import ChatOpenAI
    from langchain.agents import initialize_agent, AgentType

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    tools = [ocean_profile_analysis_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    query = (
        f"Analyze the ocean profile in {csv_path}. "
        "Is the water column stable and where is the thermocline?"
    )

    response = agent.invoke(query)
    print("\nAgent Response:\n", response)
