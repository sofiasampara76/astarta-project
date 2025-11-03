"""
Example FastAPI endpoints demonstrating bottleneck visualizations.
Install dependencies: pip install fastapi uvicorn matplotlib seaborn plotly
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from typing import Optional
import pandas as pd

from api import ingest_astarta, compute_stage_durations
from visualizations import generate_all_visualizations, prepare_bottleneck_data, STAGE_COLORS

app = FastAPI(title="Astarta Bottleneck Analysis API")

# Default stages (matching notebook)
DEFAULT_STAGES = {
    'Вїзд': ('Диспетчер вїзду', 'Початок відбору проб'),
    'Відбір_проб': ('Початок відбору проб', 'Кінець відбору проб'),
    'Зважування': ('Зважування брутто', 'Лабораторний аналіз'),
    'Лабораторія': ('Лабораторний аналіз', 'Пункт розгрузки'),
    'Розвантаження': ('Пункт розгрузки', 'Зважування тари'),
    'Виїзд': ('Зважування тари', 'Диспетчер виїзду'),
}


@app.get("/")
async def root():
    """API documentation."""
    return {
        "message": "Astarta Bottleneck Analysis API",
        "endpoints": {
            "/analyze": "GET - Analyze data and get durations",
            "/visualizations": "GET - Get all bottleneck visualizations (base64 images)",
            "/visualizations/plotly": "GET - Get all visualizations (Plotly JSON)",
            "/stats": "GET - Get summary statistics"
        }
    }


@app.get("/analyze")
async def analyze_data(
    ttn_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    mapping_json: Optional[str] = None,
    use_norm: bool = True
):
    """
    Ingest data and compute stage durations.
    Returns durations DataFrame as JSON.
    
    Query parameters:
    - ttn_path: Path to TTN file (required)
    - stations_path: Path to stations directory or file (required)
    - mapping_json: Optional path to mapping JSON file
    - use_norm: Use normalized step names (default: True)
    """
    if not ttn_path or not stations_path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required parameters",
                "required": ["ttn_path", "stations_path"],
                "example": "/analyze?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false"
            }
        )
    try:
        data = ingest_astarta(
            ttn_path=Path(ttn_path),
            station_paths=Path(stations_path),
            mapping_json=Path(mapping_json) if mapping_json else None,
            tz="Europe/Kyiv"
        )
        
        durations = compute_stage_durations(
            data["ttn"],
            DEFAULT_STAGES,
            use_norm=use_norm
        )
        
        # Convert to JSON-serializable format
        durations_dict = durations.to_dict(orient='records')
        return {
            "status": "success",
            "durations": durations_dict,
            "row_count": len(durations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations")
async def get_visualizations(
    ttn_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    mapping_json: Optional[str] = None,
    use_norm: bool = True
):
    """
    Generate all bottleneck visualizations as base64-encoded PNG images.
    Returns JSON with image data URLs (data:image/png;base64,...).
    """
    if not ttn_path or not stations_path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required parameters",
                "required": ["ttn_path", "stations_path"],
                "example": "/visualizations?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false"
            }
        )
    try:
        data = ingest_astarta(
            ttn_path=Path(ttn_path),
            station_paths=Path(stations_path),
            mapping_json=Path(mapping_json) if mapping_json else None,
            tz="Europe/Kyiv"
        )
        
        durations = compute_stage_durations(
            data["ttn"],
            DEFAULT_STAGES,
            use_norm=use_norm
        )
        
        viz = generate_all_visualizations(
            df_durations=durations,
            df_stations=data["stations"],
            output_format='base64'
        )
        
        # Return as data URLs
        result = {}
        for key, img_base64 in viz.items():
            result[key] = f"data:image/png;base64,{img_base64}"
        
        return {
            "status": "success",
            "visualizations": result,
            "descriptions": {
                "timeline": "Daily bottleneck timeline",
                "frequency": "Bottleneck frequency (daily/weekly/monthly)",
                "summary": "Summary statistics table"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations/plotly")
async def get_visualizations_plotly(
    ttn_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    mapping_json: Optional[str] = None,
    use_norm: bool = True
):
    """
    Generate all visualizations as Plotly JSON (for interactive web charts).
    Returns JSON with Plotly figure objects.
    """
    if not ttn_path or not stations_path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required parameters",
                "required": ["ttn_path", "stations_path"],
                "example": "/visualizations/plotly?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false"
            }
        )
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise HTTPException(status_code=500, detail="plotly not installed")
    
    try:
        data = ingest_astarta(
            ttn_path=Path(ttn_path),
            station_paths=Path(stations_path),
            mapping_json=Path(mapping_json) if mapping_json else None,
            tz="Europe/Kyiv"
        )
        
        durations = compute_stage_durations(
            data["ttn"],
            DEFAULT_STAGES,
            use_norm=use_norm
        )
        
        viz = generate_all_visualizations(
            df_durations=durations,
            df_stations=data["stations"],
            output_format='plotly'
        )
        
        # Convert Plotly figures to JSON-compatible format
        import json
        import numpy as np
        
        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable types to standard Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            else:
                return obj
        
        result = {}
        for key, fig in viz.items():
            try:
                # Get the figure as dict
                fig_dict = fig.to_dict()
                # Convert all numpy arrays and other types to standard Python types
                result[key] = convert_to_serializable(fig_dict)
            except Exception as e:
                result[key] = {"error": f"Failed to serialize: {str(e)}"}
        
        return {
            "status": "success",
            "visualizations": result,
            "note": "Use Plotly.js or plotly Python library to render these figures"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(
    ttn_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    mapping_json: Optional[str] = None,
    use_norm: bool = True
):
    """
    Get bottleneck statistics (matching notebook insights).
    """
    if not ttn_path or not stations_path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required parameters",
                "required": ["ttn_path", "stations_path"],
                "example": "/stats?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false"
            }
        )
    try:
        data = ingest_astarta(
            ttn_path=Path(ttn_path),
            station_paths=Path(stations_path),
            mapping_json=Path(mapping_json) if mapping_json else None,
            tz="Europe/Kyiv"
        )
        
        durations = compute_stage_durations(
            data["ttn"],
            DEFAULT_STAGES,
            use_norm=use_norm
        )
        
        # Get unique step names from data for debugging
        step_col = "step_norm" if use_norm and ("step_norm" in data["ttn"].columns) else "step_name_raw"
        unique_steps = data["ttn"][step_col].dropna().unique().tolist() if step_col in data["ttn"].columns else []
        expected_steps = set([step for stage in DEFAULT_STAGES.values() for step in stage])
        
        # Check if durations are empty
        if durations.empty or len(durations) == 0:
            return {
                "status": "warning",
                "message": "No duration data computed. Check if stage names match the data.",
                "durations_count": 0,
                "ttn_rows": len(data["ttn"]),
                "debug_info": {
                    "using_column": step_col,
                    "unique_steps_in_data": unique_steps[:20],  # First 20 for debugging
                    "expected_steps": list(expected_steps),
                    "matching_steps": [s for s in unique_steps if s in expected_steps][:10]
                },
                "insights": None,
                "overall_stats": []
            }
        
        bottleneck_data = prepare_bottleneck_data(durations)
        
        # Check if bottleneck data is empty
        overall = bottleneck_data['overall']
        if overall.empty or len(overall) == 0:
            return {
                "status": "warning",
                "message": "No bottleneck data available. Check if stage names match the data.",
                "durations_count": len(durations),
                "insights": None,
                "overall_stats": []
            }
        
        # Compute insights (matching notebook Cell 24)
        daily = bottleneck_data['daily']
        bottlenecks_daily = daily[daily['is_bottleneck'] == True] if 'is_bottleneck' in daily.columns else pd.DataFrame()
        
        # Get most common bottleneck
        most_common = bottlenecks_daily['stage'].value_counts().head(1) if len(bottlenecks_daily) > 0 else pd.Series(dtype=int)
        
        # Get statistics with safe checks
        insights = {}
        if len(overall) > 0:
            try:
                slowest_idx = overall['median'].idxmax()
                slowest = overall.loc[slowest_idx]
                insights["slowest_stage"] = {
                    "stage": slowest['stage'],
                    "median_minutes": float(slowest['median'])
                }
            except (ValueError, KeyError):
                insights["slowest_stage"] = None
            
            try:
                most_variable_idx = overall['std'].idxmax()
                most_variable = overall.loc[most_variable_idx]
                insights["most_variable_stage"] = {
                    "stage": most_variable['stage'],
                    "std_dev_minutes": float(most_variable['std'])
                }
            except (ValueError, KeyError):
                insights["most_variable_stage"] = None
            
            try:
                lowest_throughput_idx = overall['throughput_per_hour'].idxmin()
                lowest_throughput = overall.loc[lowest_throughput_idx]
                insights["lowest_throughput_stage"] = {
                    "stage": lowest_throughput['stage'],
                    "vehicles_per_hour": float(lowest_throughput['throughput_per_hour'])
                }
            except (ValueError, KeyError):
                insights["lowest_throughput_stage"] = None
        else:
            insights["slowest_stage"] = None
            insights["most_variable_stage"] = None
            insights["lowest_throughput_stage"] = None
        
        # Most common bottleneck
        if len(most_common) > 0 and len(bottlenecks_daily) > 0:
            unique_dates = len(bottlenecks_daily['date'].unique()) if 'date' in bottlenecks_daily.columns else 0
            insights["most_common_bottleneck"] = {
                "stage": most_common.index[0],
                "days": int(most_common.values[0]),
                "percentage": float(most_common.values[0] / unique_dates * 100) if unique_dates > 0 else 0.0
            }
        else:
            insights["most_common_bottleneck"] = None
        
        return {
            "status": "success",
            "insights": insights,
            "overall_stats": bottleneck_data['overall'].to_dict(orient='records'),
            "durations_count": len(durations),
            "daily_bottlenecks_count": len(bottlenecks_daily)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/view", response_class=HTMLResponse)
async def view_visualizations(
    ttn_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    mapping_json: Optional[str] = None,
    use_norm: bool = True
):
    """HTML page that displays interactive Plotly visualizations."""
    if not ttn_path or not stations_path:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualization Viewer - Missing Parameters</title>
        </head>
        <body>
            <h1>Missing Required Parameters</h1>
            <p>Please provide ttn_path and stations_path as query parameters.</p>
            <p>Example: <a href="/view?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false">/view?ttn_path=...&stations_path=...</a></p>
            <p><a href="/demo">Go to Demo Page</a></p>
        </body>
        </html>
        """
    
    # For now, return a page that will fetch the data via JavaScript
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Astarta Bottleneck Visualizations</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #4ECDC4; }
            .chart-container { margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 4px; }
            .loading { text-align: center; padding: 40px; color: #666; }
            .error { color: red; padding: 20px; background: #ffe6e6; border-radius: 4px; margin: 20px 0; }
            h2 { color: #2a3f5f; border-bottom: 2px solid #4ECDC4; padding-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Astarta Bottleneck Analysis Visualizations</h1>
            <div id="loading" class="loading">Loading visualizations...</div>
            <div id="error" class="error" style="display: none;"></div>
            <div id="charts" style="display: none;">
                <div class="chart-container" id="timeline-div">
                    <h2>1. Daily Bottleneck Timeline</h2>
                    <div id="timeline"></div>
                </div>
                <div class="chart-container" id="frequency-div">
                    <h2>2. Bottleneck Frequency</h2>
                    <div id="frequency"></div>
                </div>
                <div class="chart-container" id="summary-div">
                    <h2>3. Summary Statistics Table</h2>
                    <div id="summary"></div>
                </div>
            </div>
        </div>
        
        <script>
            const params = new URLSearchParams(window.location.search);
            const ttnPath = params.get('ttn_path') || '""" + ttn_path + """';
            const stationsPath = params.get('stations_path') || '""" + stations_path + """';
            const mappingJson = params.get('mapping_json') || '""" + (mapping_json or '') + """';
            const useNorm = params.get('use_norm') !== 'false';
            
            const apiUrl = `/visualizations/plotly?ttn_path=${encodeURIComponent(ttnPath)}&stations_path=${encodeURIComponent(stationsPath)}&use_norm=${useNorm}` + 
                          (mappingJson ? `&mapping_json=${encodeURIComponent(mappingJson)}` : '');
            
            fetch(apiUrl)
                .then(response => response.json())
                .then(data => {
                    if (data.status !== 'success') {
                        throw new Error(data.detail || 'Failed to load visualizations');
                    }
                    
                    const charts = data.visualizations;
                    const loading = document.getElementById('loading');
                    const error = document.getElementById('error');
                    const chartsDiv = document.getElementById('charts');
                    
                    loading.style.display = 'none';
                    
                    // Render each visualization
                    if (charts.timeline) {
                        Plotly.newPlot('timeline', charts.timeline.data, charts.timeline.layout);
                    }
                    if (charts.frequency) {
                        Plotly.newPlot('frequency', charts.frequency.data, charts.frequency.layout);
                    }
                    if (charts.summary) {
                        Plotly.newPlot('summary', charts.summary.data, charts.summary.layout);
                    }
                    
                    chartsDiv.style.display = 'block';
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    const errorDiv = document.getElementById('error');
                    errorDiv.textContent = 'Error loading visualizations: ' + error.message;
                    errorDiv.style.display = 'block';
                    console.error(error);
                });
        </script>
    </body>
    </html>
    """


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple HTML form page for testing endpoints."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Astarta Bottleneck Analysis Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            form { background: #f5f5f5; padding: 20px; border-radius: 8px; max-width: 600px; }
            label { display: block; margin-top: 10px; font-weight: bold; }
            input[type="text"], input[type="checkbox"] { width: 100%; padding: 8px; margin-top: 5px; }
            button { background: #4ECDC4; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-top: 15px; }
            button:hover { background: #3ba99f; }
            .endpoint { margin-top: 20px; }
            .endpoint a { color: #4ECDC4; text-decoration: none; }
            .endpoint a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Astarta Bottleneck Analysis API</h1>
        <p>Enter your data paths and test the endpoints:</p>
        
        <form id="apiForm">
            <label for="ttn_path">TTN Path:</label>
            <input type="text" id="ttn_path" name="ttn_path" value="data/Ввіз 01.06.2025- 30.10.2025.xls" required>
            
            <label for="stations_path">Stations Path:</label>
            <input type="text" id="stations_path" name="stations_path" value="data/stations_data" required>
            
            <label for="mapping_json">Mapping JSON (optional):</label>
            <input type="text" id="mapping_json" name="mapping_json" value="">
            
            <label>
                <input type="checkbox" id="use_norm" name="use_norm"> Use normalized step names (requires mapping JSON)
            </label>
            <p style="font-size: 11px; color: #666; margin-top: 5px;">
                Note: If you don't have a mapping JSON file, leave this unchecked. 
                It will use raw step names from the data.
            </p>
            
            <div class="endpoint">
                <button type="button" onclick="testEndpoint('/analyze')">Test /analyze</button>
                <button type="button" onclick="testEndpoint('/stats')">Test /stats</button>
                <button type="button" onclick="testEndpoint('/visualizations')">Test /visualizations</button>
                <button type="button" onclick="viewVisualizations()" style="background: #8F00FF;">📊 View Interactive Charts</button>
            </div>
        </form>
        
        <div id="result" style="margin-top: 20px; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 4px; display: none;">
            <h3>Result:</h3>
            <pre id="resultContent"></pre>
        </div>
        
        <div class="endpoint" style="margin-top: 30px;">
            <h3>Quick Links:</h3>
            <ul>
                <li><a href="/docs" target="_blank">Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/" target="_blank">API Root</a></li>
                <li><strong><a href="/view?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false" target="_blank">📊 View Interactive Visualizations (Recommended!)</a></strong></li>
            </ul>
        </div>
        
        <script>
            function testEndpoint(endpoint) {
                const form = document.getElementById('apiForm');
                const formData = new FormData(form);
                const params = new URLSearchParams();
                
                params.append('ttn_path', formData.get('ttn_path'));
                params.append('stations_path', formData.get('stations_path'));
                if (formData.get('mapping_json')) {
                    params.append('mapping_json', formData.get('mapping_json'));
                }
                params.append('use_norm', formData.get('use_norm') ? 'true' : 'false');
                
                const url = endpoint + '?' + params.toString();
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                resultDiv.style.display = 'block';
                resultContent.textContent = 'Loading...';
                
                fetch(url)
                    .then(response => response.json())
                    .then(data => {
                        resultContent.textContent = JSON.stringify(data, null, 2);
                    })
                    .catch(error => {
                        resultContent.textContent = 'Error: ' + error.message;
                    });
            }
            
            function viewVisualizations() {
                const form = document.getElementById('apiForm');
                const formData = new FormData(form);
                const params = new URLSearchParams();
                
                params.append('ttn_path', formData.get('ttn_path'));
                params.append('stations_path', formData.get('stations_path'));
                if (formData.get('mapping_json')) {
                    params.append('mapping_json', formData.get('mapping_json'));
                }
                params.append('use_norm', formData.get('use_norm') ? 'true' : 'false');
                
                window.location.href = '/view?' + params.toString();
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

