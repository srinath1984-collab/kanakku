import numpy as np
import json
import asyncio
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from vertexai.generative_models import GenerativeModel, GenerationConfig

class Categorizer:
    def __init__(self, user_email, db_client):
        self.user_email = user_email
        self.db = db_client
        # Tier 2 Engine: Runs locally on your Cloud Run instance
        self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Tier 3 Engine: Gemini Experts
        self.llm = GenerativeModel("gemini-2.0-flash-lite")
        self.rag_threshold = 0.95

    async def process_csv_batch(self, raw_descriptions, user_categories):
        """
        The main entry point for your /upload route.
        Returns a list of categorized transaction objects.
        """
        results = [None] * len(raw_descriptions)
        pending_llm_indices = []

        # 0. Load User's Personal Knowledge Base
        smart_rules = await self._get_smart_rules()
        vector_lib = await self._get_vector_library()

        # 1 & 2. Attempt Tier 1 (Exact) and Tier 2 (Semantic)
        for i, desc in enumerate(raw_descriptions):
            clean_desc = desc.lower().strip()

            # Tier 1: Exact Match
            if clean_desc in smart_rules:
                results[i] = {"raw_desc": desc, "category": smart_rules[clean_desc], "method": "Smart Rule"}
                continue

            # Tier 2: Semantic Match (RAG)
            rag_match = self._find_vector_match(clean_desc, vector_lib)
            if rag_match:
                results[i] = {"raw_desc": desc, "category": rag_match, "method": "RAG Match"}
                continue

            # If both fail, queue for LLM
            pending_llm_indices.append(i)

        # 3. Tier 3: Gemini Batch Call (for all remaining transactions)
        if pending_llm_indices:
            llm_input_map = {str(i): raw_descriptions[i] for i in pending_llm_indices}
            llm_results = await self._call_gemini_batch(llm_input_map, user_categories)
            
            for i_str, category in llm_results.items():
                idx = int(i_str)
                results[idx] = {"raw_desc": raw_descriptions[idx], "category": category, "method": "LLM Reasoning"}

        return results

    def _find_vector_match(self, text, library):
        if not library: return None
        target_v = self.vector_model.encode(text)
        
        best_sim = 0
        best_cat = None
        for entry in library:
            lib_v = np.array(entry['vector'])
            similarity = 1 - cosine(target_v, lib_v)
            if similarity > best_sim:
                best_sim = similarity
                best_cat = entry['category']
        
        return best_cat if best_sim >= self.rag_threshold else None

    async def _call_gemini_batch(self, keyed_descriptions, categories):
        """Sends a single JSON map to Gemini to categorize multiple items at once."""
        config = GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    k: {"type": "string", "enum": categories + ["Other", "Income", "Excluded"]}
                    for k in keyed_descriptions.keys()
                }
            }
        )
        
        prompt = f"Categorize these transactions into exactly one of {categories}. Return a JSON map with the same keys."
        try:
            response = await self.llm.generate_content_async(
                [prompt, json.dumps(keyed_descriptions)], 
                generation_config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini Batch Error: {e}")
            # Fallback to 'Other' for this chunk if the API fails
            return {k: "Other" for k in keyed_descriptions.keys()}

    async def _get_smart_rules(self):
        docs = self.db.collection("users").document(self.user_email).collection("smart_rules").stream()
        return {doc.id: doc.to_dict()['category'] for doc in docs}

    async def _get_vector_library(self):
        docs = self.db.collection("users").document(self.user_email).collection("vector_library").stream()
        return [doc.to_dict() for doc in docs]
