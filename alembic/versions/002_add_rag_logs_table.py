"""Add rag_logs table

Revision ID: 002
Revises: 001
Create Date: 2025-12-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create rag_logs table for RAG pipeline analytics."""
    op.create_table('rag_logs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('conversation_id', sa.String(36), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('query_length', sa.Integer(), nullable=False),
        sa.Column('query_intent', sa.String(length=50), nullable=True, comment='Detected intent: factual, comparative, temporal, etc.'),
        sa.Column('query_entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Extracted entities from query'),
        sa.Column('chunks_retrieved', sa.Integer(), nullable=False, comment='Number of chunks retrieved'),
        sa.Column('chunks_used', sa.Integer(), nullable=False, comment='Number of chunks actually used in prompt'),
        sa.Column('retrieval_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Similarity scores for retrieved chunks'),
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Source documents/meetings used'),
        sa.Column('answer_length', sa.Integer(), nullable=True),
        sa.Column('output_format', sa.String(length=50), nullable=True, comment='Requested output format: summary, decisions, etc.'),
        sa.Column('confidence_score', sa.Float(), nullable=True, comment='Overall confidence score (0-1)'),
        sa.Column('confidence_level', sa.String(length=20), nullable=True, comment='high, medium, low, very_low, none'),
        sa.Column('confidence_reason', sa.Text(), nullable=True),
        sa.Column('embedding_latency_ms', sa.Float(), nullable=True),
        sa.Column('retrieval_latency_ms', sa.Float(), nullable=True),
        sa.Column('generation_latency_ms', sa.Float(), nullable=True),
        sa.Column('total_latency_ms', sa.Float(), nullable=True),
        sa.Column('cache_hit', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('cache_type', sa.String(length=20), nullable=True, comment='semantic, exact, embedding, none'),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('error_type', sa.String(length=100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('user_rating', sa.Integer(), nullable=True, comment='1-5 star rating or thumbs up/down (1/0)'),
        sa.Column('user_feedback', sa.Text(), nullable=True, comment='Optional text feedback'),
        sa.Column('feedback_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('model_used', sa.String(length=100), nullable=True, comment='LLM model used for generation'),
        sa.Column('embedding_model', sa.String(length=100), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}', comment='Additional metadata, filters used, etc.'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for common query patterns
    op.create_index('ix_rag_logs_cache_hit', 'rag_logs', ['cache_hit'], unique=False)
    op.create_index('ix_rag_logs_confidence_level', 'rag_logs', ['confidence_level'], unique=False)
    op.create_index('ix_rag_logs_conversation_id', 'rag_logs', ['conversation_id'], unique=False)
    op.create_index('ix_rag_logs_created_at', 'rag_logs', ['created_at'], unique=False)
    op.create_index('ix_rag_logs_query_intent', 'rag_logs', ['query_intent'], unique=False)
    op.create_index('ix_rag_logs_success', 'rag_logs', ['success'], unique=False)
    op.create_index('ix_rag_logs_user_id', 'rag_logs', ['user_id'], unique=False)


def downgrade() -> None:
    """Drop rag_logs table."""
    op.drop_index('ix_rag_logs_user_id', table_name='rag_logs')
    op.drop_index('ix_rag_logs_success', table_name='rag_logs')
    op.drop_index('ix_rag_logs_query_intent', table_name='rag_logs')
    op.drop_index('ix_rag_logs_created_at', table_name='rag_logs')
    op.drop_index('ix_rag_logs_conversation_id', table_name='rag_logs')
    op.drop_index('ix_rag_logs_confidence_level', table_name='rag_logs')
    op.drop_index('ix_rag_logs_cache_hit', table_name='rag_logs')
    op.drop_table('rag_logs')
