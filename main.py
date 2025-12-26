from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.cloud import firestore
from google.oauth2 import id_token # Needed for verification
from google.auth.transport import requests as google_requests # Needed for verification
import pandas as pd
import io
import hashlib

app = FastAPI()
db = firestore.Client(database="kanakku")

# Replace with your actual Google Client ID from GCP Console
GOOGLE_CLIENT_ID = "940001353287-3mu3k6jd76haav5dn6tfemu46mk76dnt.apps.googleusercontent.com"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORY_MAP = {
    'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe'],
    'Transport': ['uber', 'ola', 'petrol'],
    # ... add the rest of your map here ...
}

# --- NEW: Helper function to verify the user ---
def verify_user(token: str):
    try:
        # This decodes the Google Token and gets the user's email
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        return idinfo['email']
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid login token")

def categorize_expense(description):
    desc = str(description).lower()
    for category, keywords in CATEGORY_MAP.items():
        for key in keywords:
            if key in desc:
                return category
    return 'Other'
    
@app.post("/upload")
async def upload_expenses(files: list[UploadFile] = File(...), authorization: str = Header(None)):
    # 1. Get the user email from the token
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")
    
    token = authorization.split(" ")[1] # Gets the token after "Bearer"
    user_email = verify_user(token)

    all_dataframes = []

    for file in files:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 1. Detect Columns
        desc_col = next((c for c in ['description', 'narration', 'remarks'] if c in df.columns), None)
        debit_col = next((c for c in ['debit', 'withdrawal', 'dr'] if c in df.columns), None)
        credit_col = next((c for c in ['credit', 'deposit', 'cr'] if c in df.columns), None)
        df['category'] = df[desc_col].apply(categorize_expense) if desc_col in df.columns else 'Other'
        
        
        # 2. Now user_email is defined, so we can save to Firestore
        # 1. Get the user document reference
        db.collection("users").document(user_email).set({"active": True}, merge=True)
        user_doc_ref = db.collection("users").document(user_email)

        # 2. Explicitly "Set" the user document (this removes the Italics)
        # Using merge=True ensures you don't overwrite existing settings
        user_doc_ref.set({"last_login": firestore.SERVER_TIMESTAMP,
                         "last_active": firestore.SERVER_TIMESTAMP,
                         "email": user_email
                         }, merge=True)
        user_ref = user_doc_ref.collection("expenses")
        for _, row in df.iterrows():
            # Get numeric values using the helper function
            debit_val = self_clean_float(row.get(debit_col))
            credit_val = self_clean_float(row.get(credit_col))
    
            amt = 0.0
            
            # 2. Determine Expense Amount
            # If there is a Debit value, that is our primary expense
            if debit_val > 0:
                amt = debit_val
            # If there is a negative Credit value, it's often a charge/expense
            elif credit_val < 0:
                amt = credit_val
            tx_id = generate_tx_id(user_email, row.get('date'), row[desc_col], amt)
            user_ref.document(tx_id).set({
                "description": row.get(desc_col, ""),
                "amount": amt,
                "category": row.get('category', 'Other'),
                "date": row.get('date', ""),
                "created_at": firestore.SERVER_TIMESTAMP
            }, merge=True)
        all_dataframes.append(df)

    final_df = pd.concat(all_dataframes, ignore_index=True)
    stream = io.StringIO()
    final_df.to_csv(stream, index=False)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=kanakku_report.csv"}
    )

@app.get("/summary")
async def get_summary(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")
        
    token = authorization.split(" ")[1]
    user_email = verify_user(token)
    
    expenses = db.collection("users").document(user_email).collection("expenses").stream()
    print(f"DEBUG: Starting retrieval for {user_email}")    
    summary = {}
    count = 0
    for exp in expenses:
        count = count + 1
        data = exp.to_dict()
        cat = data.get('category', 'Other')
        amt = data.get('amount', 0)
        summary[cat] = summary.get(cat, 0) + amt
    print(f"DEBUG: Successfully retrieved {count} expense documents for {user_email}")
    return summary
    
def self_clean_float(val):
    """Utility to turn messy CSV strings into clean floats."""
    if pd.isna(val) or val == "": return 0.0
    try:
        if isinstance(val, str):
            # Remove currency symbols, commas, and spaces
            s = val.replace('₹', '').replace(',', '').replace('$','').strip()
            return float(s) if s else 0.0
        return float(val)
    except:
        return 0.0
        
def generate_tx_id(user_email, date, description, amount):
    # Create a unique string for the transaction
    raw_str = f"{user_email}|{date}|{description}|{amount}"
    return hashlib.md5(raw_str.encode()).hexdigest()
