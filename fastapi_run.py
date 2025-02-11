from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
import asyncio
from collections import Counter

app = FastAPI()
csv_file = "data/detection_log.csv"

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_line_count = 0

    while True:
        df = pd.read_csv(csv_file)

        # Only send new data
        if len(df) > last_line_count:
            new_data = df.iloc[last_line_count:]  # Get all new rows since last read
            last_line_count = len(df)  # Update last read count

            # Summary Stats
            total_detections = len(df)
            total_violations = df[df["Restricted Area Violation"] == "Yes"].shape[0]
            most_frequent_class = Counter(df["Class"]).most_common(1)[0][0]
            top_5_violations = df[df["Restricted Area Violation"] == "Yes"].tail(5).to_dict(orient="records")

            # Prepare data
            data = {
                "timestamp": new_data["Timestamp"].tolist(),  # Convert to list
                "class": new_data["Class"].tolist(),  # Convert to list
                "confidence": new_data["Confidence"].apply(lambda x: round(float(x) * 100, 2)).tolist(),  # Convert to list
                "restricted_area_violation": new_data["Restricted Area Violation"].tolist(),
                "summary": {
                    "total_detections": total_detections,
                    "total_violations": total_violations,
                    "most_frequent_class": most_frequent_class,
                    "top_5_violations": top_5_violations
                }
            }

            await websocket.send_json(data)

        await asyncio.sleep(1)  # Check every second



@app.get("/data", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})

@app.websocket("/ws/data")
async def websocket_data_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        df = pd.read_csv(csv_file)

        # Ensure the Timestamp column is in datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Sort the DataFrame by Timestamp in descending order (most recent first)
        df = df.sort_values(by='Timestamp', ascending=False)

        # Convert the 'Timestamp' column to string to make it JSON serializable
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Convert the DataFrame to a list of dictionaries
        data_list = df.to_dict(orient="records")

        # Send the full data
        await websocket.send_json({"data": data_list})

        await asyncio.sleep(1)  # Send data every second
