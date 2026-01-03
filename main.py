from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Form
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
import asyncio
from polarity_computer import PolarityComputer  # <--- Import your separate file

app = FastAPI()
db = firestore.Client(database="kanakku")
# Initialize Vertex AI (Cloud Run handles credentials automatically)
vertexai.init(project="kanakku-477505", location="us-central1")
model = GenerativeModel("gemini-2.5-flash-lite")
computer = PolarityComputer() # Initialize it once

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

DEFAULT_PREFS = {
    "currency": "$",
    "dayfirst": False,
    "categories": [
        "Food", "Transport", "Shopping", "Bills", 
        "Entertainment", "Income", "Subscription", 
        "Fuel", "Excluded", "Other"
    ]
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
    
# Initialize semaphore (e.g., limit to 5 concurrent LLM calls)
llm_semaphore = asyncio.Semaphore(5)
async def categorize_with_llm_async(descriptions, user_categories):
    if not descriptions:
        return []
    async with llm_semaphore:
        keyed_input = {str(i): desc for i, desc in enumerate(descriptions)}
        # Inject the user's specific categories into the prompt
        # Ensure 'Excluded' is in the list sent to the LLM even if it's hidden in the UI
        if "Excluded" not in user_categories:
            user_categories.append("Excluded")
        category_list_str = ", ".join(user_categories)
        print(f"DEBUG: categories for gemini are {category_list_str}")
        system_instruction = f"""
        You are a precision financial classifier. 
        TASK: Map each transaction ID to the MOST LIKELY category from this list: {category_list_str}.
        
        RULES:
        1. You MUST return a category for EVERY key provided.
        2. Only use 'Other' if there is absolutely no semantic match.
        3. Use 'Income' for salaries, rewards, or refunds.
        4. Use 'Excluded' for credit card payments or transfers.
        5. ONLY use categories from the provided list. If no match exists, use 'Other'. DO NOT invent new categories.
        
        OUTPUT: Return a JSON object where the keys match the input IDs.
        Example: {{"0": "Food", "1": "Transport"}}. 
        
        Transactions to categorize:
        {json.dumps(keyed_input)}
    
        SPECIAL RULE: If a transaction looks like a credit card payment, 
        a transfer between accounts, or a self-payment, use the 'Excluded' category.
        Examples: "CC PAYMENT", "AUTOPAY", "TRANSFER TO SAVINGS", "ONLINE PAYMENT".
        """
    
        # 2. Initialize the model WITH the instructions
        model = GenerativeModel(
            "gemini-2.5-flash-lite",
            system_instruction=[system_instruction] # Must be a list or Content object
        )
        
        try:
            response = await model.generate_content_async(
                json.dumps(keyed_input),
                generation_config={"response_mime_type": "application/json"}
            )
            
            # 2. Parse the dictionary back into a list in the CORRECT order
            res_dict = json.loads(response.text)
            
            # Reconstruct list by checking every index to guarantee length
            final_list = []
            for i in range(len(descriptions)):
                final_list.append(res_dict.get(str(i), "Other"))
                
            return final_list
            
        except Exception as e:
            print(f"LLM Error: {e}")
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
    try:
        user_email = verify_user(authorization.split(" ")[1]).lower().strip()
        
        query = db.collection("users").document(user_email).collection("expenses")
        query = query.where("month_key", "==", month)
        
        # Handle the case where categories might be an empty string
        if categories and categories.strip():
            cat_list = [c.strip() for c in categories.split(",") if c.strip()]
            if cat_list:
                query = query.where("category", "in", cat_list)
        
        # If order_by causes a 500, it's usually a missing index. 
        # Let's try fetching without sorting first to verify data exists.
        docs = query.stream()
        
        results = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            results.append(data)
            
        # Sort manually in Python to avoid needing a complex Firestore Index for now
        results.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return results

    except Exception as e:
        print(f"DRILLDOWN ERROR: {str(e)}")
        # Returning a proper HTTPException ensures CORS headers are still sent
        raise HTTPException(status_code=500, detail=str(e))

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
        return {**DEFAULT_PREFS, **doc.to_dict()}
    # Default values
    return DEFAULT_PREFS

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

MAX_FILE_SIZE = 2 * 1024 * 1024 # 2MB Limit
MAX_ROWS = 2000 # Max transactions per upload

def normalize_row(row, debit_col, credit_col, amt_col, mode):
    """
    Normalizes a row into a consistent (amount, is_income) tuple.
    Modes: 'standard' (Pos=Income), 'inverted' (Pos=Expense)
    """
    # 1. Handle Explicit Columns (Highest Priority)
    if debit_col and credit_col:
        d_val = self_clean_float(row.get(debit_col))
        c_val = self_clean_float(row.get(credit_col))
        if c_val != 0: return abs(c_val), True
        if d_val != 0: return abs(d_val), False

    # 2. Handle Single Amount Column
    raw_amt = self_clean_float(row.get(amt_col))
    if mode == 'inverted':
        # Pos is Debit (Expense), Neg is Credit (Income)
        return abs(raw_amt), raw_amt < 0
    else:
        # Standard: Pos is Credit (Income), Neg is Debit (Expense)
        return abs(raw_amt), raw_amt > 0
        
@app.post("/upload")
async def upload_expenses(files: list[UploadFile] = File(...), authorization: str = Header(None)):
    print("DEBUG: 1. Upload endpoint hit")
    token = authorization.split(" ")[1]
    user_email = verify_user(token).lower().strip()
    
    prefs_doc = db.collection("users").document(user_email).collection("settings").document("preferences").get()
    if prefs_doc.exists:
        data = prefs_doc.to_dict()
        dayfirst_pref = data.get('dayfirst', DEFAULT_PREFS["dayfirst"])
        user_categories = data.get('categories', DEFAULT_PREFS["categories"])
    else:
        # Fallback for brand new users
        dayfirst_pref = DEFAULT_PREFS["dayfirst"]
        user_categories = DEFAULT_PREFS["categories"]
    
    all_rows = []

    for file in files:
        content = await file.read() # Read content
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File {file.filename} is too large (Max 2MB).")
        df = pd.read_csv(io.BytesIO(content))
        if len(df) > MAX_ROWS:
            raise HTTPException(status_code=413, detail=f"CSV exceeds limit of {MAX_ROWS} transactions.")
        df.columns = [c.lower().strip() for c in df.columns]
        print(f"DEBUG: 2. File read. Rows in CSV: {len(df)}")
        print(f"DEBUG: 2c. Columns in CSV: {df.columns}")
        
        # Enhanced Detection: Look for 'amount' if debit/credit aren't found
        desc_col = next((c for c in ['description', 'narration', 'remarks', 'details'] if c in df.columns), None)
        debit_col = next((c for c in ['debit', 'withdrawal', 'dr'] if c in df.columns), None)
        credit_col = next((c for c in ['credit', 'deposit', 'cr'] if c in df.columns), None)
        amt_col = next((c for c in ['amount', 'value', 'transaction amount'] if c in df.columns), None)
        date_col = next((c for c in ['date', 'posting date'] if c in df.columns), None)

        effective_mode = await computer.compute_mode(df, debit_col, credit_col, amt_col)
        for _, row in df.iterrows():
            is_income = False
            final_amt = 0.0

            if effective_mode == "explicit":
                d_val = self_clean_float(row.get(debit_col))
                c_val = self_clean_float(row.get(credit_col))
                if c_val != 0:
                    final_amt, is_income = abs(c_val), True
                elif d_val != 0:
                    final_amt, is_income = abs(d_val), False
            
            else:
                raw_amt = self_clean_float(row.get(amt_col))
                if raw_amt == 0: continue
                
                if effective_mode == "inverted":
                    # Mode: Positive is Expense (Standard US Credit Card style)
                    is_income = raw_amt < 0
                else:
                    # Mode: Positive is Income (Standard Bank style)
                    is_income = raw_amt > 0
                
                final_amt = abs(raw_amt)

            if final_amt == 0: continue
            date = str(row.get(date_col, ''))
            all_rows.append({
                "raw_desc": str(row.get(desc_col, "Unknown")),
                "amount": final_amt,
                "is_income": is_income,
                "date": str(row.get(date_col, ''))
            })
    print(f"DEBUG 3: Done parsing {len(all_rows)}")

    if not all_rows:
        print("DEBUG: 4. ERROR: No rows to process")
        return {"status": "error", "message": "No valid transactions found in CSV."}

  # --- THE MOST LIKELY CRASH POINT ---
    print("DEBUG: 5. Calling Gemini LLM...")
    try:
        descriptions = [r['raw_desc'] for r in all_rows]
        chunk_size = 30
        chunks = [descriptions[i:i + chunk_size] for i in range(0, len(descriptions), chunk_size)]
        print(f"DEBUG: Parallel processing {len(chunks)} chunks...")
        # FIRE ALL REQUESTS AT ONCE
        tasks = [categorize_with_llm_async(chunk, user_categories) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
    
        # Flatten the list of lists into one single list of categories
        llm_categories = [item for sublist in chunk_results for item in sublist]
        
        #llm_categories = categorize_with_llm(descriptions, user_categories)
        print(f"DEBUG: 6. Gemini returned {len(llm_categories)} categories")
    except Exception as e:
        print(f"DEBUG: 6. ERROR: Gemini Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Categorization Failed: {str(e)}")

    # 3. Save to Firestore
    print("DEBUG: 7. Starting Firestore Writes...")
    batch = db.batch()
    count = 0
    for i, row in enumerate(all_rows):
        std_date = parse_to_standard_date(row['date'], dayfirst_pref)
        m_key = parse_to_month_key(row['date'], dayfirst_pref)
        final_cat = "Income" if row['is_income'] else llm_categories[i]
        #print(f"DEBUG: 7A final_cat = {llm_categories[i]} for desc {row['raw_desc']}")
        
        tx_id = generate_tx_id(user_email, std_date, row['raw_desc'], row['amount'])
        doc_ref = db.collection("users").document(user_email).collection("expenses").document(tx_id)
        # Add operation to the current batch
        batch.set(doc_ref, {
            "description": row['raw_desc'],
            "amount": row['amount'],
            "category": final_cat,
            "date": std_date,
            "month_key": m_key,
            "created_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
    
        count += 1

        # THE CRITICAL TRIGGER: Every 500 operations, commit and start fresh
        if count % 500 == 0:
            print(f"DEBUG: Committing batch of 500 (Total: {count})", flush=True)
            batch.commit()
            batch = db.batch() # Re-initialize for the next 500 items
    # FINAL COMMIT: Send the remaining items (e.g., the last 423 items)
    if count % 500 != 0:
        print(f"DEBUG: Committing final batch of {count % 500} items", flush=True)
        batch.commit()
    print(f"DEBUG: 8 Successfully wrote {count} documents to Firestore.", flush=True)
    
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
    total_income = 0
    total_expense = 0
    for exp in expenses:
        count = count + 1
        data = exp.to_dict()
        cat = data.get('category', 'Other')
        if cat == "Excluded":
            continue
        amt = data.get('amount', 0)
        summary[cat] = summary.get(cat, 0) + amt
        # 1. Total Income logic (typically 'Income' category)
        if cat == "Income":
            total_income += amt
        
        # 2. Total Expense logic (Exclude 'Excluded' and 'Income')
        elif cat != "Excluded":
            total_expense += amt
            # Keep category breakdown for the list
    print(f"DEBUG: Breakdown = {summary}")
    return {
        "breakdown": summary,
        "total_income": total_income,
        "total_expense": total_expense
    } 
