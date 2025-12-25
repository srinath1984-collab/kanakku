from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.cloud import firestore
import pandas as pd
import io

app = FastAPI()

# Initialize the client for your specific database
db = firestore.Client(database="kanakku")

# CRITICAL: This allows your frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default category map
CATEGORY_MAP = {
    'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe', 'starbucks', 'mcdonalds'],
    'Groceries': ['blinkit', 'zepto', 'bigbasket', 'supermarket', 'mart'],
    'Transport': ['uber', 'ola', 'rapido', 'petrol', 'fuel', 'metro'],
    'Shopping': ['amazon', 'flipkart', 'myntra', 'zara', 'h&m'],
    'Bills & Utilities': ['jio', 'airtel', 'bescom', 'recharge', 'insurance'],
    'Entertainment': ['netflix', 'prime video', 'hotstar', 'cinema', 'bookmyshow']
}

def categorize_expense(description):
    """Checks the description against our keyword map."""
    desc = str(description).lower()
    for category, keywords in CATEGORY_MAP.items():
        for key in keywords:
            if key in desc:
                return category
    return 'Other' # Fallback if no keywords match
    
@app.post("/upload")
async def upload_expenses(files: list[UploadFile] = File(...)):
    all_dataframes = []

    for file in files:
        # Read the file (works for CSV, can be extended for XLS)
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Clean up column names (standardize to lowercase)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 2. Apply categorization
        # Assumes your CSV has a column called 'description' or 'narration'
        desc_col = 'description' if 'description' in df.columns else 'narration'
        
        if desc_col in df.columns:
            df['category'] = df[desc_col].apply(categorize_expense)
        else:
            df['category'] = 'Unknown Column'
        user_ref = db.collection("users").document(user_email).collection("expenses")
        for _, row in df.iterrows():
            user_ref.add({
                "description": row.get(desc_col, ""),
                "amount": float(row.get('amount', 0)),
                "category": row.get('category', 'Other'),
                "date": row.get('date', ""),
                "created_at": firestore.SERVER_TIMESTAMP
            })

        all_dataframes.append(df)

    # 3. Merge all files into one
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Convert to CSV for return
    stream = io.StringIO()
    final_df.to_csv(stream, index=False)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=kanakku_report.csv"}
    )
    
@app.get("/summary")
async def get_summary(authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    user_email = verify_user(token)
    
    expenses = db.collection("users").document(user_email).collection("expenses").stream()
    
    summary = {}
    for exp in expenses:
        data = exp.to_dict()
        cat = data.get('category', 'Other')
        amt = data.get('amount', 0)
        summary[cat] = summary.get(cat, 0) + amt
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
