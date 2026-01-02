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
        Determine the 'mode':
        - 'standard': One amount column. Positive is Income (Salary/Refund).
        - 'inverted': One amount column. Positive is Expense (Amazon/Uber).
        
        CSV Sample:
        {csv_text}
        
        Return ONLY a JSON object: {{"mode": "standard" | "inverted"}}
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
