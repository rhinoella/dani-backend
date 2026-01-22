from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, computed_field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    APP_NAME: str = "DANI Engine"
    ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Ollama Configuration
    # OLLAMA_ENV controls LLM only - embeddings ALWAYS use local Ollama
    OLLAMA_ENV: str = "local"  
    OLLAMA_LOCAL_URL: str = "http://localhost:11434"
    OLLAMA_CLOUD_URL: str = "https://ollama.com"
    OLLAMA_API_KEY: Optional[str] = None  # Required when OLLAMA_ENV="cloud"
    
    @computed_field
    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Compute Ollama URL for LLM based on environment setting.
        
        Note: This is used by the LLM client. For cloud mode, user needs
        to ensure the cloud URL supports their chosen model.
        """
        return self.OLLAMA_CLOUD_URL if self.OLLAMA_ENV == "cloud" else self.OLLAMA_LOCAL_URL
    
    @computed_field
    @property
    def OLLAMA_EMBEDDINGS_URL(self) -> str:
        """Embeddings URL - ALWAYS local Ollama.
        
        Cloud Ollama does not support embeddings API, so embeddings
        always use the local Ollama instance.
        """
        return self.OLLAMA_LOCAL_URL
    
    LLM_MODEL: str = "ministral-3:8b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    
    LLM_NUM_CTX: int = 8192      # Increased from 4096 for better context
    LLM_NUM_PREDICT: int = 2048
    LLM_NUM_THREAD: int = 8      # Increased from 4 for better performance
    LLM_TEMPERATURE: float = 0.7

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_TRANSCRIPTS: str = "meeting_transcripts"
    QDRANT_COLLECTION_DOCUMENTS: str = "documents"
    QDRANT_COLLECTION_EMAIL_STYLES: str = "email_styles"

    DB_CONNECTION: str = "postgresql+asyncpg"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_DATABASE: str = "dani_db"
    DB_USERNAME: str = "dani"
    DB_PASSWORD: str = "dani_secret_2024"
    
    DATABASE_POOL_SIZE: int = 20      # Increased from 10 for more concurrent users
    DATABASE_MAX_OVERFLOW: int = 50   # Increased from 20 for traffic spikes
    DATABASE_POOL_TIMEOUT: int = 30
    
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"{self.DB_CONNECTION}://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_DATABASE}"
    
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50   # Increased from 20 for better caching
    REDIS_SOCKET_TIMEOUT: float = 5.0
    REDIS_SOCKET_CONNECT_TIMEOUT: float = 5.0
    
    GOOGLE_CLIENT_ID: str = "__MISSING__"
    GOOGLE_AUTH_ENABLED: bool = True
    
    JWT_SECRET_KEY: str = "__MISSING__"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 50   # Increased from 20 for more requests
    RATE_LIMIT_PER_DAY: int = 1000    # Increased from 500 for more daily usage
    
    MAX_CONVERSATIONS_PER_USER: int = 500   # Increased from 100 for more conversations
    MAX_MESSAGES_PER_CONVERSATION: int = 500  # Increased from 200 for longer conversations
    MAX_MESSAGE_LENGTH: int = 8000           # Increased from 4000 for longer messages
    
    MIN_HISTORY_MESSAGES: int = 6
    MAX_HISTORY_MESSAGES: int = 10     # Reduced for faster retrieval
    SUMMARIZE_AFTER_MESSAGES: int = 20
    CONTEXT_BUDGET_TOKENS: int = 4000  # Increased from 2000 for longer conversations
    TOKENS_PER_MESSAGE_ESTIMATE: int = 150
    CONTEXT_TOKEN_BUDGET: int = 2000
    SUMMARIZE_THRESHOLD: int = 20

    FIREFLIES_API_KEY: str = "__MISSING__"
    FIREFLIES_BASE_URL: str = "https://api.fireflies.ai/graphql"
    FIREFLIES_WEBHOOK_SECRET: str = "__MISSING__"
    
    MAX_TRANSCRIPT_SIZE_MB: float = 10.0
    MAX_CHUNKS_PER_TRANSCRIPT: int = 1000
    MAX_QUERY_LENGTH: int = 10000
    MAX_BATCH_SIZE: int = 100   # Increased from 50 for faster bulk operations
    
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_TTL_SECONDS: int = 1800
    SEMANTIC_CACHE_MAX_SIZE: int = 500   # Increased from 200 for better caching
    
    EMBEDDING_CACHE_ENABLED: bool = True
    EMBEDDING_CACHE_SIMILARITY_THRESHOLD: float = 0.98
    EMBEDDING_CACHE_TTL_SECONDS: int = 3600
    EMBEDDING_CACHE_MAX_SIZE: int = 1000  # Increased from 500 for better caching
    
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_VECTOR_WEIGHT: float = 0.5
    HYBRID_KEYWORD_WEIGHT: float = 0.5
    
    ADAPTIVE_RETRIEVAL_ENABLED: bool = True
    ADAPTIVE_MIN_SIMILARITY: float = 0.10  # Lowered from 0.35 - current embeddings give ~0.15-0.18 scores
    ADAPTIVE_MAX_CHUNKS: int = 25
    ADAPTIVE_MIN_CHUNKS: int = 5
    
    RERANKING_ENABLED: bool = True
    CROSS_ENCODER_ENABLED: bool = False
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Enhanced RAG settings for 90%+ accuracy
    ENHANCED_RETRIEVAL_ENABLED: bool = True  # Enable multi-stage retrieval
    QUERY_EXPANSION_ENABLED: bool = True  # Generate query variants
    QUERY_EXPANSION_VARIANTS: int = 2  # Number of query variants
    CONTEXTUAL_COMPRESSION_ENABLED: bool = True  # Compress chunks to relevant parts
    LLM_RERANKING_ENABLED: bool = True  # Use LLM for re-ranking
    MIN_RETRIEVAL_RELEVANCE: float = 0.60  # Minimum score after re-ranking
    
    # Enhanced memory settings
    SEMANTIC_MEMORY_SEARCH_ENABLED: bool = True  # Search past messages semantically
    ENTITY_EXTRACTION_ENABLED: bool = True  # Extract entities from conversations
    TOPIC_SUMMARY_ENABLED: bool = True  # Generate topic summaries
    
    LLM_WARMUP_ENABLED: bool = True
    TIMING_DETAILED: bool = True
    
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "dani-documents"
    S3_ENDPOINT_URL: Optional[str] = None
    S3_DOCUMENTS_PREFIX: str = "documents/"
    S3_PRESIGNED_URL_EXPIRY: int = 3600
    
    MCP_ENABLED: bool = True
    GEMINI_API_KEY: str = ""
    IMGBB_API_KEY: str = ""
    MCP_GENERATED_IMAGES_DIR: str = "./generated_images"
    MCP_CONNECTION_TIMEOUT: float = 30.0
    MCP_TOOL_TIMEOUT: float = 60.0
    MCP_MAX_RETRIES: int = 3
    MCP_RETRY_DELAY: float = 1.0
    MCP_AUTO_RECONNECT: bool = True
    MCP_HEALTH_CHECK_INTERVAL: int = 60
    MCP_MAX_CONCURRENT_CALLS: int = 10   # Increased from 5 for better tool usage
    MCP_ALLOWED_COMMANDS: str = "npx,uvx,node,python"

    @field_validator("FIREFLIES_API_KEY")
    @classmethod
    def validate_fireflies_key(cls, v: str) -> str:
        if not v or v == "__MISSING__":
            import os
            if os.getenv("ENV", "development") != "development":
                raise ValueError("FIREFLIES_API_KEY is required and was not provided")
            return "__MISSING__"
        return v
    
    @field_validator("GOOGLE_CLIENT_ID")
    @classmethod
    def validate_google_client_id(cls, v: str) -> str:
        if not v or v == "__MISSING__":
            import os
            if os.getenv("ENV", "development") != "development":
                raise ValueError("GOOGLE_CLIENT_ID is required for production")
            return "__MISSING__"
        return v


settings = Settings()
