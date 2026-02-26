CONTEXT_PRUNING_PROMPT = """You are a Knowledge Graph Engineer.
    
New Raw Notes from a research agent:
{text_block} 

Your task is to:
1. Extract all atomic, verifiable facts from the New Raw Notes.
2. For each fact, identify its source URL.
3. Assign a confidence score (1-100) based on the credibility of the source.
4. Ignore any information that is the agent's internal "thinking" or planning.

Return ONLY a valid JSON object with a single key 'new_facts' containing a list of these structured facts.
"""
