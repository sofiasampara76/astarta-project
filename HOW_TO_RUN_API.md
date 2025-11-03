# How to Test and Run the API

## Prerequisites

Install required packages:

```powershell
.\.venv\Scripts\python -m pip install fastapi uvicorn matplotlib seaborn plotly pandas openpyxl xlrd
```

## Step 1: Test the API Components

Before starting the server, verify everything works:

```powershell
.\.venv\Scripts\python test_api.py
```

This will test:

- ✓ Data ingestion (TTN + stations)
- ✓ Duration computation
- ✓ Visualization generation

Expected output:

```
============================================================
ASTARTA API TEST SUITE
============================================================
============================================================
Test 1: Data Ingestion
============================================================
✓ TTN rows: 47,500
✓ Stations rows: 51,055
...
✓ ALL TESTS PASSED!
```

## Step 2: Start the API Server

### Option A: Direct Python

```powershell
.\.venv\Scripts\python example_api_endpoints.py
```

### Option B: Using uvicorn (recommended)

```powershell
.\.venv\Scripts\python -m uvicorn example_api_endpoints:app --reload --host 0.0.0.0 --port 8000
```

The server will start at: `http://localhost:8000`

## Step 3: Test API Endpoints

### 1. Check API is running

Open in browser: `http://localhost:8000`

Or use curl:

```powershell
curl http://localhost:8000
```

### 2. Test `/analyze` endpoint (GET - can open in browser!)

```powershell
# PowerShell syntax
Invoke-RestMethod -Uri "http://localhost:8000/analyze?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false"

# Or just open in browser:
# http://localhost:8000/analyze?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false
```

### 3. Test `/stats` endpoint

```powershell
curl "http://localhost:8000/stats?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false"
```

### 4. Test `/visualizations` endpoint (base64 images)

```powershell
curl "http://localhost:8000/visualizations?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false" -o response.json
```

The response will contain base64-encoded images in the `visualizations` field.

### 5. Test `/visualizations/plotly` endpoint (interactive charts)

```powershell
curl "http://localhost:8000/visualizations/plotly?ttn_path=data/Ввіз%2001.06.2025-%2030.10.2025.xls&stations_path=data/stations_data&use_norm=false" -o plotly_response.json
```

### 6. View API documentation

Open in browser: `http://localhost:8000/docs`

This shows an interactive Swagger UI where you can test all endpoints.

## Available Endpoints

| Endpoint                 | Method | Description                                 |
| ------------------------ | ------ | ------------------------------------------- |
| `/`                      | GET    | API documentation                           |
| `/analyze`               | GET    | Analyze data and get durations              |
| `/visualizations`        | GET    | Get all visualizations as base64 PNG images |
| `/visualizations/plotly` | GET    | Get all visualizations as Plotly JSON       |
| `/stats`                 | GET    | Get bottleneck statistics                   |
| `/demo`                  | GET    | Simple HTML demo page                       |
| `/docs`                  | GET    | Interactive API documentation (Swagger UI)  |

## Visualizations Generated

1. **timeline** - Daily bottleneck timeline (which stage was bottleneck each day)
2. **heatmap** - Time on each stage by day (heatmap)
3. **frequency** - Bottleneck frequency (daily/weekly/monthly percentages)
4. **throughput** - Throughput timeseries by station
5. **queue_pressure** - Queue pressure between LAB → UNLOAD over time
6. **summary** - Summary statistics table

## Troubleshooting

### Error: "Missing optional dependency 'xlrd'"

```powershell
.\.venv\Scripts\python -m pip install xlrd
```

### Error: "No module named 'fastapi'"

```powershell
.\.venv\Scripts\python -m pip install fastapi uvicorn
```

### Error: "matplotlib or plotly required"

```powershell
.\.venv\Scripts\python -m pip install matplotlib seaborn plotly
```

### Path encoding issues in URLs

Use URL encoding for paths with spaces:

- `Ввіз 01.06.2025- 30.10.2025.xls` → `Ввіз%2001.06.2025-%2030.10.2025.xls`

Or use the `/docs` endpoint (Swagger UI) which handles this automatically.

## Example: Using from Python

```python
import requests

# Get stats
response = requests.get(
    "http://localhost:8000/stats",
    params={
        "ttn_path": "data/Ввіз 01.06.2025- 30.10.2025.xls",
        "stations_path": "data/stations_data",
        "use_norm": False
    }
)
stats = response.json()
print(stats["insights"])
```

## PowerShell vs curl

**Note:** In PowerShell, `curl` is an alias for `Invoke-WebRequest`. Use:

- `Invoke-RestMethod` - for GET requests (auto-parses JSON)
- `Invoke-WebRequest` - for any request (returns full response object)
- Or use `curl.exe` (if curl is installed) for curl-style syntax
- Or use Python `requests` library (recommended for complex requests)
