"""
Adaptive Ocean Profile Analysis Tool
------------------------------------
Agent-callable pipeline for analyzing ERDDAP / ARGO vertical profile CSVs.

✔ Depth-adaptive (0–N meters)
✔ Query-aware physics
✔ Scientifically honest outputs
✔ Safe for partial & deep profiles
"""

from typing import Dict, Optional, Tuple, Union, List, Any
import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import gsw


# ============================================================
# 0. Physics helpers
# ============================================================
def pressure_to_depth(pressure_dbar: np.ndarray, latitude: float) -> np.ndarray:
    """Convert pressure (dbar) to depth (meters) using TEOS-10."""
    return -gsw.z_from_p(pressure_dbar, latitude)


# ============================================================
# 1. Quality Control
# ============================================================
def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality control to ocean profile data.
    
    Args:
        df: Input DataFrame containing ocean profile data
        
    Returns:
        DataFrame with quality-controlled data
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = {"pres", "temp", "psal"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
        
    df = df.copy()

    # Convert to numeric, coercing errors to NaN
    for col in ["pres", "temp", "psal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply quality filters
    filtered = df[
        (df["pres"] > 0) &
        (df["temp"].between(-2, 40)) &
        (df["psal"].between(0, 42)) &
        (df["pres"].notna()) &
        (df["temp"].notna()) &
        (df["psal"].notna())
    ]
    
    if filtered.empty:
        raise ValueError("No valid data points after quality control")

    return filtered.sort_values("pres").reset_index(drop=True)


# ============================================================
# 2. Vertical Binning
# ============================================================
def bin_vertical_profile(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """Bin vertical profile data into pressure levels.
    
    Args:
        df: Input DataFrame with pressure, temperature, and salinity data
        step: Pressure bin size in dbar (default: 10)
        
    Returns:
        DataFrame with binned and averaged data
        
    Raises:
        ValueError: If input data is empty or invalid
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    if step <= 0:
        raise ValueError("Step size must be positive")
        
    df = df.copy()
    df["pres_bin"] = (df["pres"] // step) * step

    # Group by pressure bin and calculate mean of all numeric columns
    binned = df.groupby("pres_bin", as_index=False).mean(numeric_only=True)
    
    if binned.empty:
        raise ValueError("No valid data after binning")
        
    return binned


# ============================================================
# 3. Smoothing
# ============================================================
def smooth_profile(
    df: pd.DataFrame,
    window: int = 7,
    polyorder: int = 2
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to temperature and salinity profiles.
    
    Args:
        df: Input DataFrame with pressure, temperature, and salinity data
        window: Size of the smoothing window (must be odd)
        polyorder: Order of the polynomial used for smoothing
        
    Returns:
        DataFrame with smoothed temperature and salinity profiles
        
    Raises:
        ValueError: If input data is invalid or smoothing fails
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    if window < 3:
        raise ValueError("Window size must be at least 3")
        
    if polyorder < 1:
        raise ValueError("Polynomial order must be at least 1")

    df = df.copy()

    if len(df) < window:
        window = max(3, len(df) // 2 * 2 + 1)
    
    df["temp_smooth"] = savgol_filter(df["temp"], window, polyorder)
    df["psal_smooth"] = savgol_filter(df["psal"], window, polyorder)

    return df


# ============================================================
# 4. Derived Physics (TEOS-10)
# ============================================================
def compute_physics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lat = df["latitude"].iloc[0]
    lon = df["longitude"].iloc[0]

    df["depth_m"] = pressure_to_depth(df["pres_bin"].values, lat)

    SA = gsw.SA_from_SP(df["psal_smooth"], df["pres_bin"], lon, lat)
    CT = gsw.CT_from_t(SA, df["temp_smooth"], df["pres_bin"])

    df["density_sigma0"] = gsw.sigma0(SA, CT)

    # Correct gradients (with respect to depth)
    df["dT_dz"] = np.gradient(df["temp_smooth"], df["depth_m"])
    df["dS_dz"] = np.gradient(df["psal_smooth"], df["depth_m"])

    return df


# ============================================================
# 5. Depth Context
# ============================================================
def compute_depth_context(df: pd.DataFrame) -> Dict:
    min_d = float(df["depth_m"].min())
    max_d = float(df["depth_m"].max())

    return {
        "min_depth_m": min_d,
        "max_depth_m": max_d,
        "is_shallow_profile": max_d < 300,
        "is_mid_depth_profile": 300 <= max_d < 1000,
        "is_deep_profile": max_d >= 1000
    }


# ============================================================
# 6. Thermocline Detection (Adaptive)
# ============================================================
def detect_thermocline(
    df: pd.DataFrame,
    depth_context: Dict
) -> Optional[float]:

    max_depth = depth_context["max_depth_m"]
    upper_limit = min(500, 0.4 * max_depth)

    candidate = df[
        (df["depth_m"] > 10) &
        (df["depth_m"] < upper_limit)
    ]

    if len(candidate) < 5:
        return None

    idx = candidate["dT_dz"].abs().idxmax()
    return float(df.loc[idx, "depth_m"])


# ============================================================
# 7. Layer Classification (Adaptive)
# ============================================================
def classify_layers(
    df: pd.DataFrame,
    thermocline_depth_m: Optional[float],
    depth_context: Dict
) -> pd.DataFrame:

    df = df.copy()

    if thermocline_depth_m is None:
        df["layer"] = "Observed Water Column"
        return df

    if depth_context["is_shallow_profile"]:
        conditions = [
            df["depth_m"] <= 10,
            df["depth_m"] > 10
        ]
        labels = ["Surface Layer", "Subsurface Layer"]

    else:
        conditions = [
            df["depth_m"] <= 10,
            (df["depth_m"] > 10) & (df["depth_m"] <= thermocline_depth_m),
            df["depth_m"] > thermocline_depth_m
        ]
        labels = ["Surface Layer", "Thermocline", "Deep Ocean"]

    df["layer"] = np.select(conditions, labels, default="Unclassified")
    return df


# ============================================================
# 8. Insight Extraction
# ============================================================
def extract_insights(df: pd.DataFrame) -> Dict:
    depth_context = compute_depth_context(df)
    thermocline_depth_m = detect_thermocline(df, depth_context)
    # Stability (aggregate, tolerant)
    density_diff = np.diff(df["density_sigma0"])
    stable_fraction = float(np.mean(density_diff > 0))
    df = classify_layers(df, thermocline_depth_m, depth_context)
    layer_stats = {}
    for layer in df["layer"].unique():
        d = df[df["layer"] == layer]
        layer_stats[layer.lower().replace(" ", "_")] = {
            "min_depth": float(d["depth_m"].min()),
            "max_depth": float(d["depth_m"].max()),
            "avg_temp": float(d["temp_smooth"].mean()),
            "avg_salinity": float(d["psal_smooth"].mean())
        }
    return {
        "depth_context": depth_context,
        "thermocline_depth_m": thermocline_depth_m,
        "stable_fraction": stable_fraction,
        "is_stratified": stable_fraction > 0.8,
        "surface_temp": float(df["temp_smooth"].iloc[0]),
        "temperature_at_max_depth": float(df["temp_smooth"].iloc[-1]),
        "surface_salinity": float(df["psal_smooth"].iloc[0]),
        "salinity_at_max_depth": float(df["psal_smooth"].iloc[-1]),
        "max_observed_depth_m": depth_context["max_depth_m"],
        "layers": layer_stats,
        "depth_coverage_note": (
            f"Analysis limited to {depth_context['max_depth_m']:.1f} m "
            "based on ERDDAP query constraints."
        )
    }


# ============================================================
# 9. Single Profile Pipeline
# ============================================================
def analyze_single_profile(
    profile_df: pd.DataFrame,
    bin_step: int = 10,
    smooth_window: int = 7
) -> Optional[Dict]:

    df = quality_control(profile_df)
    if len(df) < 6:
        return None

    df = bin_vertical_profile(df, bin_step)
    df = smooth_profile(df, smooth_window)
    df = compute_physics(df)

    insights = extract_insights(df)
    if insights is None:
        return None

    insights.update({
        "profile_id": profile_df["profile_id"].iloc[0],
        "latitude": df["latitude"].iloc[0],
        "longitude": df["longitude"].iloc[0],
        "time": profile_df["time"].iloc[0] if "time" in profile_df else None
    })

    return insights


# ============================================================
# 10. Multi-profile Entry Point
# ============================================================
def analyze_ocean_profile(
    csv_path: str,
    bin_step: int = 10,
    smooth_window: int = 7
) -> Dict[str, Dict[str, Any]]:
    """Analyze multiple ocean profiles from a CSV file.
    
    Args:
        csv_path: Path to CSV file containing ocean profile data
        bin_step: Vertical bin size in dbar (default: 10)
        smooth_window: Smoothing window size (default: 7)
        
    Returns:
        Dictionary containing analysis results for each profile
        
    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If CSV file is empty or invalid
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
        
    if bin_step <= 0:
        raise ValueError("Bin step must be positive")
        
    if smooth_window < 1:
        raise ValueError("Smoothing window must be at least 1")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    df = df.rename(columns={"platform_number": "profile_id"})
    required = ["profile_id", "latitude", "longitude", "pres", "temp", "psal"]
    df = df.dropna(subset=required)

    results = {}
    for pid, group in df.groupby("profile_id"):
        insights = analyze_single_profile(group, bin_step, smooth_window)
        if insights:
            results[pid] = insights

    return {"profiles": results}


# ============================================================
# 11. LangChain Tool Wrapper
# ============================================================
from langchain.tools import tool

@tool
def ocean_profile_analysis_tool(
    csv_path: Optional[str] = None,
    bin_step: int = 10,
    smooth_window: int = 7
) -> Dict[str, Dict[str, Any]]:
    """LangChain tool for analyzing ocean profile data.
    
    Args:
        csv_path: Path to CSV file containing ocean profile data.
                 If None, will look for ARGO data in the default location.
        bin_step: Vertical bin size in dbar (default: 10)
        smooth_window: Smoothing window size (default: 7)
        
    Returns:
        Dictionary containing analysis results for each profile
        
    Example:
        >>> results = ocean_profile_analysis_tool("path/to/data.csv")
    """
    if csv_path is None:
        # Look for ARGO data in the data directory relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, "..", "data", "argo_data.csv")
        if os.path.exists(default_path):
            csv_path = default_path
        else:
            raise ValueError(
                "No CSV path provided and default ARGO data not found. "
                "Please specify a CSV file path."
            )
    """
    Analyze ERDDAP / ARGO ocean profiles and return
    depth-adaptive, physics-based insights.
    """
    return analyze_ocean_profile(csv_path, bin_step, smooth_window)


def main() -> None:
    """Command-line interface for ocean profile analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze ocean profile data from CSV files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the CSV file containing ocean profile data',
    )
    parser.add_argument(
        '--bin_step',
        type=int,
        default=10,
        help='Vertical bin size in dbar',
        metavar='N'
    )
    parser.add_argument(
        '--smooth_window',
        type=int,
        default=7,
        help='Smoothing window size (must be odd)',
        metavar='W'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: print to stdout)',
        metavar='FILE'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isfile(args.csv_file):
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
        
    if args.bin_step <= 0:
        print("Error: bin_step must be positive", file=sys.stderr)
        sys.exit(1)
        
    if args.smooth_window < 3 or args.smooth_window % 2 == 0:
        print("Error: smooth_window must be an odd number >= 3", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Analyze the profile
        results = analyze_ocean_profile(
            args.csv_file,
            bin_step=args.bin_step,
            smooth_window=args.smooth_window
        )
        
        output = []
        
        # Generate output for each profile
        for profile_id, insights in results["profiles"].items():
            profile_output = [
                f"\n=== Profile ID: {profile_id} ===",
                f"Latitude: {insights['latitude']:.4f}",
                f"Longitude: {insights['longitude']:.4f}"
            ]
            
            if insights.get('time'):
                profile_output.append(f"Timestamp: {insights['time']}")
            
            profile_output.extend([
                "\nTemperature:",
                f"  Surface: {insights['surface_temp']:.2f}°C",
                f"  Bottom: {insights['temperature_at_max_depth']:.2f}°C",
                "\nSalinity:",
                f"  Surface: {insights['surface_salinity']:.2f} PSU",
                f"  Bottom: {insights['salinity_at_max_depth']:.2f} PSU"
            ])
            
            if insights['thermocline_depth_m']:
                profile_output.extend([
                    "\nThermocline:",
                    f"  Depth: {insights['thermocline_depth_m']:.1f} m"
                ])
            
            profile_output.extend([
                f"\nWater Column Stability: {'Stable' if insights['is_stratified'] else 'Unstable'}",
                "\nWater Column Layers:"
            ])
            
            for layer, stats in insights['layers'].items():
                profile_output.extend([
                    f"  - {layer.replace('_', ' ').title()}:",
                    f"    Depth: {stats['min_depth']:.1f} - {stats['max_depth']:.1f} m",
                    f"    Avg Temp: {stats['avg_temp']:.2f}°C",
                    f"    Avg Salinity: {stats['avg_salinity']:.2f} PSU"
                ])
            
            profile_output.append(f"\n{insights['depth_coverage_note']}")
            output.append("\n".join(profile_output))
        
        # Write to file or print to stdout
        result_text = "\n\n".join(output)
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    f.write(result_text)
                print(f"Results saved to {args.output}")
            except IOError as e:
                print(f"Error writing to {args.output}: {e}", file=sys.stderr)
                print("\nResults:")
                print(result_text)
        else:
            print(result_text)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()