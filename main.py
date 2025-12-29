from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.cloud import firestore
from google.oauth2 import id_token # Needed for verification
from google.auth.transport import requests as google_requests # Needed for verification
from dateutil import parser as date_parser
import pandas as pd
import io
import hashlib
import vertexai
from vertexai.generative_models import GenerativeModel
import json
from pydantic import BaseModel

app = FastAPI()
db = firestore.Client(database="kanakku")
# Initialize Vertex AI (Cloud Run handles credentials automatically)
vertexai.init(project="kanakku-477505", location="us-central1")
model = GenerativeModel("gemini-2.5-flash-lite")

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

def parse_to_month_key(date_str, dayfirst_pref):
    try:
        # dateutil.parser automatically detects formats like 12-03-2025 or 2025/03/12
        # dayfirst=True is helpful for Indian/UK formats (DD-MM-YYYY)
        dt = date_parser.parse(str(date_str), dayfirst=dayfirst_pref)
        
        # Returns standardized "2025-03" regardless of input format
        return dt.strftime("%Y-%m")
    except Exception as e:
        print(f"Date Parsing Error: {e} for string {date_str}")
        return "unknown"

def parse_to_standard_date(date_str, dayfirst_pref):
    try:
        dt = date_parser.parse(str(date_str), dayfirst=dayfirst_pref)
        return dt.strftime("%Y-%m-%d") # Standardizes the full date for deduplication
    except:
        return str(date_str)

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

def categorize_with_llm(descriptions, user_categories):
    if not descriptions:
        return []

    # Inject the user's specific categories into the prompt
    category_list_str = ", ".join(user_categories)
    system_instruction = f"""
    You are a financial assistant. Categorize these transactions into EXACTLY ONE 
    of these user-defined categories: {category_list_str}, or 'Other'.
    Return ONLY a JSON list of strings in the same order as the input. No markdown, no explanation.
    
    Examples:
    "UBER PENDING" -> "Transport"
    "ZOMATO*RESTAURANT" -> "Food"
    "REVERSAL-FEE" -> "Income"
    
    Transactions to categorize:
    {json.dumps(descriptions)}
    """

    response = model.generate_content(
            f"Categorize: {json.dumps(descriptions)}",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )    
    # Clean potential markdown wrapping from LLM response
    clean_text = response.text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except:
        # Fallback if JSON parsing fails
        return ["Other"] * len(descriptions)
        
class UpdateTx(BaseModel):
    doc_id: str
    category: str

@app.post("/update-transaction")
async def update_transaction(data: UpdateTx, authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    
    # Path to the specific transaction document
    doc_ref = db.collection("users").document(user_email).collection("expenses").document(data.doc_id)
    
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Transaction not found")
        
    # Update only the category field
    doc_ref.update({"category": data.category})
    
    return {"status": "success"}
    
@app.get("/drilldown")
async def get_drilldown(month: str, categories: str = None, authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    
    query = db.collection("users").document(user_email).collection("expenses")
    query = query.where("month_key", "==", month)
    
    # categories is a comma-separated string from the frontend
    if categories:
        cat_list = categories.split(",")
        # Use 'in' operator for multiple categories
        query = query.where("category", "in", cat_list)
    
    docs = query.order_by("date", direction=firestore.Query.DESCENDING).stream()
    
    results = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id  # <--- CRUCIAL: Add the Firestore ID to the dict
        results.append(data)
    return results    

@app.get("/analytics")
async def get_analytics(authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    
    # Fetch all expenses for this user
    docs = db.collection("users").document(user_email).collection("expenses").stream()
    
    # We want a structure like: {"2025-01": {"Food": 500, "Transport": 200}, "2025-02": {...}}
    history = {}
    for doc in docs:
        d = doc.to_dict()
        m = d.get('month_key', 'unknown')
        cat = d.get('category', 'Other')
        amt = d.get('amount', 0)
        
        if m not in history: history[m] = {}
        history[m][cat] = history.get(m).get(cat, 0) + amt
        
    # Sort by month key so the charts flow correctly
    sorted_history = dict(sorted(history.items()))
    return sorted_history
    
@app.get("/download")
async def download_csv(month: str, authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    
    # Query only the data for the specific month
    docs = db.collection("users").document(user_email).collection("expenses").where("month_key", "==", month).stream()
    
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=report_{month}.csv"}
    )
    
@app.get("/preferences")
async def get_prefs(authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    doc = db.collection("users").document(user_email).collection("settings").document("preferences").get()
    if doc.exists:
        return doc.to_dict()
    # Default values
    return {"currency": "₹", "dayfirst": True}

@app.post("/preferences")
async def save_prefs(prefs: dict, authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    db.collection("users").document(user_email).collection("settings").document("preferences").set(prefs)
    return {"status": "success"}
    
@app.get("/available-months")
async def get_months(authorization: str = Header(None)):
    user_email = verify_user(authorization.split(" ")[1]).lower().strip()
    
    # Fetch only the month_key field from all expenses
    docs = db.collection("users").document(user_email).collection("expenses").select(["month_key"]).stream()
    
    # Use a set to get unique keys and sort them descending
    months = sorted(list(set(d.to_dict().get('month_key') for d in docs if d.to_dict().get('month_key'))), reverse=True)
    
    return {"months": months}    
    
@app.post("/upload")
async def upload_expenses(files: list[UploadFile] = File(...), authorization: str = Header(None)):
    print("DEBUG: 1. Upload endpoint hit")
    token = authorization.split(" ")[1]
    user_email = verify_user(token).lower().strip()
    
    prefs_doc = db.collection("users").document(user_email).collection("settings").document("preferences").get()
    dayfirst_pref = prefs_doc.to_dict().get('dayfirst', True) if prefs_doc.exists else True
    prefs_data = prefs_doc.to_dict() if prefs_doc.exists else {}
    user_categories = prefs_data.get('categories', ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Income", "Subscription", "Fuel", "Other"])

    all_rows = []

    for file in files:
        content = await file.read() # Read content
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.lower().strip() for c in df.columns]
        print(f"DEBUG: 2. File read. Rows in CSV: {len(df)}")
        
        # Enhanced Detection: Look for 'amount' if debit/credit aren't found
        desc_col = next((c for c in ['description', 'narration', 'remarks', 'details'] if c in df.columns), None)
        debit_col = next((c for c in ['debit', 'withdrawal', 'dr'] if c in df.columns), None)
        credit_col = next((c for c in ['credit', 'deposit', 'cr'] if c in df.columns), None)
        amt_col = next((c for c in ['amount', 'value', 'transaction amount'] if c in df.columns), None)

        for _, row in df.iterrows():
            # Try specific columns first, then general amount column
            d_val = self_clean_float(row.get(debit_col)) if debit_col else 0.0
            c_val = self_clean_float(row.get(credit_col)) if credit_col else 0.0
            a_val = self_clean_float(row.get(amt_col)) if amt_col else 0.0
            
            # Logic: If Debit/Credit exist, use them. Otherwise use general Amount.
            if d_val != 0:
                amt = d_val
            elif c_val != 0:
                amt = -c_val # Negative means Income
            else:
                amt = a_val # Could be + or - depending on bank

            if amt == 0: continue

            all_rows.append({
                "raw_desc": str(row.get(desc_col, "Unknown")),
                "amount": abs(amt),
                "is_income": amt < 0,
                "date": str(row.get('date', ''))
            })
    print(f"DEBUG 3: Done parsing {len(all_rows)}")

    if not all_rows:
        print("DEBUG: 4. ERROR: No rows to process")
        return {"status": "error", "message": "No valid transactions found in CSV."}

  # --- THE MOST LIKELY CRASH POINT ---
    print("DEBUG: 5. Calling Gemini LLM...")
    try:
        descriptions = [r['raw_desc'] for r in all_rows]
        llm_categories = categorize_with_llm(descriptions, user_categories)
        print(f"DEBUG: 6. Gemini returned {len(llm_categories)} categories")
    except Exception as e:
        print(f"DEBUG: 6. ERROR: Gemini Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Categorization Failed: {str(e)}")

    # 3. Save to Firestore
    print("DEBUG: 7. Starting Firestore Writes...")
    user_ref = db.collection("users").document(user_email).collection("expenses")
    for i, row in enumerate(all_rows):
        std_date = parse_to_standard_date(row['date'], dayfirst_pref)
        m_key = parse_to_month_key(row['date'], dayfirst_pref)
        final_cat = "Income" if row['is_income'] else llm_categories[i]
        
        tx_id = generate_tx_id(user_email, std_date, row['raw_desc'], row['amount'])
        
        user_ref.document(tx_id).set({
            "description": row['raw_desc'],
            "amount": row['amount'],
            "category": final_cat,
            "date": std_date,
            "month_key": m_key,
            "created_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
    print("DEBUG: 8. Upload Complete")
    return {"status": "success", "count": len(all_rows)}
    
@app.get("/summary")
async def get_summary(month: str = None, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")
        
    token = authorization.split(" ")[1]
    user_email = verify_user(token)
    query = db.collection("users").document(user_email).collection("expenses")
    # If a month is provided (e.g., "2025-12"), filter by it
    if month:
        expenses = query.where("month_key", "==", month).stream()
    else:
        expenses = query.stream()   
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
 
