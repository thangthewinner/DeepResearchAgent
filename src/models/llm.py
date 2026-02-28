"""
Centralized LLM model initialization.

All models are initialized here once and imported by consumer modules.
This avoids redundant init_chat_model calls scattered across the codebase.
"""

from langchain.chat_models import init_chat_model

from config.settings import (
    BASE_MODEL,
    COMPRESS_MODEL,
    COMPRESS_MAX_TOKENS,
    COMPRESSOR_MODEL,
    CREATIVE_MODEL,
    CRITIC_MODEL,
    JUDGE_MODEL,
    OPENAI_API_KEY,
    SUMMARIZATION_MODEL,
    WRITER_MODEL,
)

# Core reasoning model (supervisor, clarification, researcher)
base_model = init_chat_model(model=BASE_MODEL, api_key=OPENAI_API_KEY)

# Creative model for draft generation
creative_model = init_chat_model(model=CREATIVE_MODEL, api_key=OPENAI_API_KEY)

# Writer model for final report
writer_model = init_chat_model(model=WRITER_MODEL, api_key=OPENAI_API_KEY)

# Compress model for researcher findings
compress_model = init_chat_model(
    model=COMPRESS_MODEL, max_tokens=COMPRESS_MAX_TOKENS, api_key=OPENAI_API_KEY
)

# Fast/cheap model for context pruning
compressor_model = init_chat_model(model=COMPRESSOR_MODEL, api_key=OPENAI_API_KEY)

# Summarization model for webpage content
summarization_model = init_chat_model(
    model=SUMMARIZATION_MODEL, api_key=OPENAI_API_KEY
)

# Adversarial critic model
critic_model = init_chat_model(model=CRITIC_MODEL, api_key=OPENAI_API_KEY)

# Quality judge model
judge_model = init_chat_model(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)
