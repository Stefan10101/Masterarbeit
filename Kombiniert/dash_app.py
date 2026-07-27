#!/usr/bin/env python3
"""
Master Dataset Dashboard - Plotly Dash Version (with Network Provider filter)
"""

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== CONFIG ====================
DATA_DIR = Path(DATA_ROOT / "kombiniert")

ALL_VARS = ["relative_humidity", "precipitation", "snow_height", "temperature", "wind_speed"]

VAR_DISPLAY = {
    "relative_humidity": "Relative humidity (%)",
    "precipitation": "Precipitation (mm)",
    "snow_height": "Snow height (cm)",
    "temperature": "Temperature (Â°C)",
    "wind_speed": "Wind speed (m/s)",
}

VAR_COLOR = {
    "relative_humidity": "#2ca02c",
    "precipitation": "#1f77b4",
    "snow_height": "#e377c2",
    "temperature": "#d62728",
    "wind_speed": "#1f1f1f",
}

TAB_LABELS = ["Alle", "Precipitation", "Snow Height", "Temperature", "Wind Speed", "Relative Humidity"]
TAB_TO_VAR = {
    "Alle": None,
    "Precipitation": "precipitation",
    "Snow Height": "snow_height",
    "Temperature": "temperature",
    "Wind Speed": "wind_speed",
    "Relative Humidity": "relative_humidity",
}

DURATIONS = {
    "3 days": timedelta(days=3),
    "7 days": timedelta(days=7),
    "14 days": timedelta(days=14),
    "30 days": timedelta(days=30),
    "60 days": timedelta(days=60),
    "90 days": timedelta(days=90),
    "91 days": timedelta(days=91),
    "92 days": timedelta(days=92),
    "100 days": timedelta(days=100),
}

DEFAULT_STATION = "Innsbruck UniversitÃ¤t"
DEFAULT_START = datetime(2020, 1, 1).date()
DEFAULT_DURATION = "3 days"

# ==================== DATA ====================
@lru_cache(maxsize=1)
def load_metadata():
    path = DATA_DIR / "stations_overview.csv"
    if not path.exists():
        raise FileNotFoundError(f"stations_overview.csv not found in {DATA_DIR}")
    df = pd.read_csv(path)
    df["var_set"] = df["variables"].str.split(", ").apply(set)
    df["n_vars"] = df["var_set"].apply(len)
    return df.sort_values("station_name").reset_index(drop=True)


def load_station_parquet(parquet_name: str, var_set: set) -> pd.DataFrame:
    path = DATA_DIR / parquet_name
    cols = ["timestamp", "station_name", "lat", "lon", "hoehe"] + list(var_set)
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


meta = load_metadata()
network_providers = sorted(meta["network_provider"].dropna().unique().tolist())

# ==================== HELPER FUNCTIONS ====================
def create_map_figure(filtered_meta: pd.DataFrame, selected_station: str | None):
    if filtered_meta.empty:
        return go.Figure()

    df = filtered_meta.copy()
    df["color"] = "#3b82f6"
    df["size"] = 6

    if selected_station and selected_station in df["station_name"].values:
        mask = df["station_name"] == selected_station
        df.loc[mask, "color"] = "#ef4444"
        df.loc[mask, "size"] = 10

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        hover_name="station_name",
        hover_data={"hoehe": ":.0f", "n_vars": True, "network_provider": True},
        color="color",
        color_discrete_map="identity",
        size="size",
        size_max=10,
        zoom=6.2,
        center={"lat": 47.05, "lon": 10.9},
        mapbox_style="open-street-map",
        height=420,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    fig.update_traces(marker=dict(opacity=0.85))
    return fig


def create_timeseries_figure(df: pd.DataFrame, station: str, hoehe: float, start, end, var: str | None):
    title = f"{station} â€¢ {hoehe:.0f} m â€¢ {start.date()} â†’ {end.date()}"

    if var is None:  # Alle
        present = [v for v in ALL_VARS if v in df.columns]
        if not present:
            fig = go.Figure()
            fig.add_annotation(text="No variables available", showarrow=False)
            return fig

        fig = make_subplots(
            rows=len(present), cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[VAR_DISPLAY[v] for v in present]
        )
        for i, v in enumerate(present, 1):
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[v], mode="lines",
                           line=dict(color=VAR_COLOR[v], width=1.5),
                           connectgaps=False),
                row=i, col=1
            )
            fig.update_yaxes(title_text=VAR_DISPLAY[v].split("(")[0].strip(), row=i, col=1)

        fig.update_layout(height=max(200 * len(present), 520), title=title,
                          margin=dict(t=50, b=30, l=50, r=20), showlegend=False)
    else:
        if var not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {VAR_DISPLAY.get(var, var)}", showarrow=False)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[var], mode="lines",
                                 line=dict(color=VAR_COLOR[var], width=1.7),
                                 connectgaps=False))
        fig.update_layout(title=title, xaxis_title="Time (UTC)",
                          yaxis_title=VAR_DISPLAY[var], height=440,
                          margin=dict(t=50, b=30, l=50, r=20), showlegend=False)

    # Dynamic y-axis with 20% padding + forced min=0 for physical variables
    def _set_yaxis_range(fig_obj, y_data, var_name, is_subplot=False, row=None):
        if len(y_data) == 0 or y_data.isna().all():
            return
        y_min = float(y_data.min())
        y_max = float(y_data.max())

        force_zero = var_name in ["snow_height", "precipitation", "wind_speed"]
        if force_zero:
            y_min = max(0.0, y_min)

        y_range = y_max - y_min
        padding = y_range * 0.20 if y_range > 0 else 1.0
        new_min = y_min - padding
        new_max = y_max + padding

        if force_zero:
            new_min = max(0.0, new_min)

        if is_subplot and row:
            fig_obj.update_yaxes(range=[new_min, new_max], row=row, col=1)
        else:
            fig_obj.update_yaxes(range=[new_min, new_max])

    if var is None:
        for i, v in enumerate(present, 1):
            _set_yaxis_range(fig, df[v], v, is_subplot=True, row=i)
    else:
        _set_yaxis_range(fig, df[var], var)

    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


# ==================== APP ====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Master Dataset Dashboard"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Master Dataset Explorer â€” 2020â€“2025", className="mb-1"), width=12),
        dbc.Col(html.P("30-min meteorological data â€¢ Tirol & surroundings", className="text-muted"), width=12),
    ], className="mb-3"),

    # Variable Tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs(
                id="variable-tabs",
                active_tab="Alle",
                children=[dbc.Tab(label=l, tab_id=l) for l in TAB_LABELS],
                className="mb-2"
            ),
        ], width=12),
    ]),

    # Time controls
    dbc.Row([
        dbc.Col([
            dbc.Label("Start date"),
            dcc.DatePickerSingle(
                id="start-date",
                date=DEFAULT_START,
                min_date_allowed=datetime(2020, 1, 1).date(),
                max_date_allowed=datetime(2025, 12, 31).date(),
                display_format="YYYY-MM-DD",
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("Duration"),
            dcc.Dropdown(
                id="duration-dropdown",
                options=[{"label": k, "value": k} for k in DURATIONS.keys()],
                value=DEFAULT_DURATION,
                clearable=False,
            ),
        ], width=2),
        dbc.Col([
            dbc.Button("Reset to defaults", id="reset-btn", color="secondary", className="mt-4"),
        ], width=2),
    ], className="mb-3"),

    # Season selector
    dbc.Row([
        dbc.Col(html.H6("Quick season selection", className="mb-1"), width=12),
        dbc.Col([
            dbc.Label("Season"),
            dcc.Dropdown(
                id="season-dropdown",
                options=[
                    {"label": "Winter (Decâ€“Feb)", "value": "winter"},
                    {"label": "Spring (Marâ€“May)", "value": "spring"},
                    {"label": "Summer (Junâ€“Aug)", "value": "summer"},
                    {"label": "Fall (Sepâ€“Nov)", "value": "fall"},
                ],
                value=None,
                placeholder="Select season...",
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("Year"),
            dcc.Dropdown(
                id="season-year-dropdown",
                options=[{"label": str(y), "value": y} for y in range(2020, 2026)],
                value=2024,
            ),
        ], width=2),
        dbc.Col([
            dbc.Button("Apply season", id="apply-season-btn", color="primary", className="mt-4"),
        ], width=2),
    ], className="mb-3 border rounded p-2 bg-light"),

    # Filters row: Elevation + Number of variables + Network Provider
    dbc.Row([
        dbc.Col([
            dbc.Label("Elevation filter"),
            dcc.Dropdown(
                id="elevation-filter",
                options=[
                    {"label": "All elevations", "value": "all"},
                    {"label": "> 3000 m", "value": ">3000"},
                    {"label": "2500 â€“ 2999 m", "value": "2500-2999"},
                    {"label": "2000 â€“ 2499 m", "value": "2000-2499"},
                    {"label": "1500 â€“ 1999 m", "value": "1500-1999"},
                    {"label": "1000 â€“ 1499 m", "value": "1000-1499"},
                    {"label": "< 1000 m", "value": "<1000"},
                ],
                value="all",
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            dbc.Label("Variables measured"),
            dcc.Dropdown(
                id="n-vars-filter",
                options=[
                    {"label": "Any number", "value": "all"},
                    {"label": "5 variables", "value": 5},
                    {"label": "4 variables", "value": 4},
                    {"label": "3 variables", "value": 3},
                    {"label": "2 variables", "value": 2},
                    {"label": "1 variable", "value": 1},
                ],
                value="all",
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            dbc.Label("Network provider"),
            dcc.Dropdown(
                id="network-filter",
                options=[{"label": p, "value": p} for p in network_providers],
                multi=True,
                value=[],
                placeholder="All providers",
            ),
        ], width=4),
    ], className="mb-2"),

    # Main content
    dbc.Row([
        dbc.Col([
            html.H5("Map"),
            dcc.Graph(
                id="map-graph",
                config={"scrollZoom": True, "displayModeBar": True}
            ),
            html.H5("Station list", className="mt-3"),
            dcc.Dropdown(id="station-dropdown", placeholder="Select or search station...", className="mb-2"),
            html.Small(id="station-count", className="text-muted"),
        ], width=5),

        dbc.Col([
            html.H5("Time series"),
            dcc.Graph(id="timeseries-graph", style={"height": "520px"}),
            html.Small(id="plot-info", className="text-muted"),
        ], width=7),
    ]),

    dbc.Row([
        dbc.Col(html.Small("Data source: Unified master dataset v2.0 (10 m clustering)"), width=12),
    ], className="mt-2 text-muted"),
], fluid=True)


# ==================== CALLBACKS ====================
@callback(
    Output("map-graph", "figure"),
    Output("station-dropdown", "options"),
    Output("station-dropdown", "value"),
    Output("station-count", "children"),
    Input("variable-tabs", "active_tab"),
    Input("station-dropdown", "value"),
    Input("elevation-filter", "value"),
    Input("n-vars-filter", "value"),
    Input("network-filter", "value"),
)
def update_map_and_stations(active_tab, current_station, elev_filter, nvars_filter, network_filter):
    var = TAB_TO_VAR.get(active_tab)

    if var is None:
        fmeta = meta.copy()
    else:
        fmeta = meta[meta["var_set"].apply(lambda s: var in s)].copy()

    # Elevation filter
    if elev_filter and elev_filter != "all":
        if elev_filter == ">3000":
            fmeta = fmeta[fmeta["hoehe"] > 3000]
        elif elev_filter == "2500-2999":
            fmeta = fmeta[(fmeta["hoehe"] >= 2500) & (fmeta["hoehe"] < 3000)]
        elif elev_filter == "2000-2499":
            fmeta = fmeta[(fmeta["hoehe"] >= 2000) & (fmeta["hoehe"] < 2500)]
        elif elev_filter == "1500-1999":
            fmeta = fmeta[(fmeta["hoehe"] >= 1500) & (fmeta["hoehe"] < 2000)]
        elif elev_filter == "1000-1499":
            fmeta = fmeta[(fmeta["hoehe"] >= 1000) & (fmeta["hoehe"] < 1500)]
        elif elev_filter == "<1000":
            fmeta = fmeta[fmeta["hoehe"] < 1000]

    # Number of variables filter
    if nvars_filter and nvars_filter != "all":
        fmeta = fmeta[fmeta["n_vars"] == int(nvars_filter)]

    # Network provider filter (multi-select)
    if network_filter and len(network_filter) > 0:
        fmeta = fmeta[fmeta["network_provider"].isin(network_filter)]

    if fmeta.empty:
        return go.Figure(), [], None, "No stations match filters"

    # Validate current station
    if current_station not in fmeta["station_name"].values:
        current_station = fmeta["station_name"].iloc[0]

    options = [{"label": s, "value": s} for s in fmeta["station_name"]]
    fig = create_map_figure(fmeta, current_station)
    count_text = f"{len(fmeta)} stations shown"

    return fig, options, current_station, count_text


@callback(
    Output("timeseries-graph", "figure"),
    Output("plot-info", "children"),
    Input("variable-tabs", "active_tab"),
    Input("station-dropdown", "value"),
    Input("start-date", "date"),
    Input("duration-dropdown", "value"),
)
def update_plot(active_tab, station, start_date, duration_label):
    if not station:
        return go.Figure(), "No station selected"

    var = TAB_TO_VAR.get(active_tab)
    row = meta[meta["station_name"] == station].iloc[0]
    pq_name = row["parquet"]
    var_set = row["var_set"]
    hoehe = row["hoehe"]

    try:
        df = load_station_parquet(pq_name, var_set)
    except Exception as e:
        return go.Figure(), f"Error loading data: {e}"

    if start_date is None:
        start_date = DEFAULT_START

    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = start_ts + pd.Timedelta(DURATIONS.get(duration_label, timedelta(days=3)))

    dff = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()

    if dff.empty:
        return go.Figure(), "No data in selected time window"

    fig = create_timeseries_figure(dff, station, hoehe, start_ts, end_ts, var)
    info = f"{len(dff):,} records shown â€¢ {pq_name}"

    return fig, info


@callback(
    Output("station-dropdown", "value", allow_duplicate=True),
    Input("map-graph", "clickData"),
    prevent_initial_call=True
)
def update_station_from_map(clickData):
    if clickData and "points" in clickData:
        station = clickData["points"][0].get("hovertext")
        if station:
            return station
    return no_update


@callback(
    Output("start-date", "date"),
    Output("duration-dropdown", "value"),
    Output("station-dropdown", "value", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_defaults(n_clicks):
    return DEFAULT_START, DEFAULT_DURATION, DEFAULT_STATION


@callback(
    Output("start-date", "date", allow_duplicate=True),
    Output("duration-dropdown", "value", allow_duplicate=True),
    Input("apply-season-btn", "n_clicks"),
    State("season-dropdown", "value"),
    State("season-year-dropdown", "value"),
    prevent_initial_call=True
)
def apply_meteorological_season(n_clicks, season, year):
    if not season or not year:
        return no_update, no_update

    year = int(year)

    if season == "winter":
        start = datetime(year - 1, 12, 1).date()
        duration = "90 days"
    elif season == "spring":
        start = datetime(year, 3, 1).date()
        duration = "92 days"
    elif season == "summer":
        start = datetime(year, 6, 1).date()
        duration = "92 days"
    elif season == "fall":
        start = datetime(year, 9, 1).date()
        duration = "91 days"
    else:
        return no_update, no_update

    return start, duration


if __name__ == "__main__":
    app.run(debug=True, port=8050)
