import pandas as pd
import json
from vertexai.generative_models import GenerativeModel, GenerationConfig

class PolarityComputer:
    def __init__(self, model_name="gemini-2.0-flash-lite"):
        self.model = GenerativeModel(model_name)

    async def compute_mode(self, df: pd.DataFrame, debit_col, credit_col, amt_col):
        # LAYER 1: THE VETO (Structural Facts)
        if debit_col and credit_col:
            print("DEBUG: Structural Veto - Explicit columns found.")
            return "explicit"

        # LAYER 2: THE LLM OPINION
        # We only send a tiny sample to keep it fast (~500ms)
        sample_csv = df.head(5).to_csv(index=False)
        llm_guess = await self._get_llm_guess(sample_csv)
        
        if llm_guess:
            print(f"DEBUG: LLM Logic - Classified as {llm_guess}")
            return llm_guess

        # LAYER 3: THE FALLBACK
        print("DEBUG: Fallback Logic - Using standard polarity.")
        return "standard"

    async def _get_llm_guess(self, csv_text):
        prompt = f"""
            Analyze these 5 rows of a bank statement.
            
            TASK: Determine the 'mode' based on merchant intent:
            - 'standard': Positive values are Income (Salary, Interest, Transfers In).
            - 'inverted': Positive values are Expenses (Shopping, Dining, Uber).
            
            CSV Sample:
            {csv_text}
            
            STEPS:
            1. Identify a clear expense (e.g., Uber, Zomato, Rent).
            2. Check its sign in the sample.
            3. If an expense is POSITIVE, the mode is 'inverted'.
            4. If an expense is NEGATIVE, the mode is 'standard'.
            
            Return ONLY a JSON object: {{"reasoning": "...", "mode": "standard" | "inverted"}}
        """
        
        config = GenerationConfig(
            temperature=0.0, 
            response_mime_type="application/json"
        )
        
        try:
            response = await self.model.generate_content_async(prompt, generation_config=config)
            res = json.loads(response.text)
            return res.get("mode")
        except Exception as e:
            print(f"LLM Precheck Error: {e}")
            return None
