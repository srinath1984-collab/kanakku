from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import io

app = FastAPI()

# CRITICAL: This allows your frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_expenses(files: list[UploadFile] = File(...)):
    combined_data = []

    for file in files:
        # Read the uploaded file (works for CSV)
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # You can add your "Categorization" logic here
        # For now, we just append them together
        combined_data.append(df)

    # Merge all dataframes into one
    final_df = pd.concat(combined_data, ignore_index=True)

    # Convert the merged dataframe back to a CSV string
    stream = io.StringIO()
    final_df.to_csv(stream, index=False)
    
    # Return the file as a downloadable response
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=combined_expenses.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
