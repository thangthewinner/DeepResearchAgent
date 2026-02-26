import os

from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Research Configuration
MAX_CONCURRENT_RESEARCHERS = 1
MAX_RESEARCHER_ITERATIONS = 3
MAX_CONTEXT_LENGTH = 250000

# Model Configuration
BASE_MODEL = "openai:gpt-4o"
CREATIVE_MODEL = "openai:gpt-4o"
SUMMARIZATION_MODEL = "openai:gpt-4o-mini"
COMPRESS_MODEL = "openai:gpt-4o"
WRITER_MODEL = "openai:gpt-5"
CRITIC_MODEL = "openai:gpt-4o"
JUDGE_MODEL = "openai:gpt-4o"
COMPRESSOR_MODEL = "openai:gpt-4o-mini"

# Token Limits
COMPRESS_MAX_TOKENS = 16000
WRITER_MAX_TOKENS = 16000
