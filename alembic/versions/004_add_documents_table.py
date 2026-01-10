"""Add documents table for file uploads

Revision ID: 004
Revises: 003
Create Date: 2025-06-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def enum_exists(enum_name: str) -> bool:
    """Check if an enum type exists in the database."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text("SELECT 1 FROM pg_type WHERE typname = :name"),
        {"name": enum_name}
    )
    return result.fetchone() is not None


def upgrade() -> None:
    """Create documents table for tracking uploaded files."""
    
    # Skip if table already exists
    if table_exists('documents'):
        return
    
    bind = op.get_bind()
    
    # Create enum types only if they don't exist
    if not enum_exists('documenttype'):
        bind.execute(sa.text(
            "CREATE TYPE documenttype AS ENUM ('pdf', 'docx', 'txt', 'markdown', 'other')"
        ))
    
    if not enum_exists('documentstatus'):
        bind.execute(sa.text(
            "CREATE TYPE documentstatus AS ENUM ('pending', 'processing', 'completed', 'failed', 'deleted')"
        ))
    
    # Create documents table using raw SQL to avoid SQLAlchemy recreating enums
    bind.execute(sa.text("""
        CREATE TABLE documents (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
            filename VARCHAR(500) NOT NULL,
            original_filename VARCHAR(500) NOT NULL,
            title VARCHAR(500),
            description TEXT,
            file_type documenttype NOT NULL,
            file_size BIGINT NOT NULL,
            mime_type VARCHAR(100),
            status documentstatus NOT NULL DEFAULT 'pending',
            error_message TEXT,
            chunk_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            collection_name VARCHAR(100) DEFAULT 'documents',
            metadata JSONB DEFAULT '{}',
            processing_started_at TIMESTAMPTZ,
            processing_completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            deleted_at TIMESTAMPTZ
        )
    """))
    
    # Create indexes for efficient querying
    op.create_index('ix_documents_user_id', 'documents', ['user_id'])
    op.create_index('ix_documents_status', 'documents', ['status'])
    op.create_index('ix_documents_file_type', 'documents', ['file_type'])
    op.create_index('ix_documents_created_at', 'documents', ['created_at'])
    op.create_index('ix_documents_deleted_at', 'documents', ['deleted_at'])
    
    # Composite index for common query patterns
    op.create_index(
        'ix_documents_user_status', 
        'documents', 
        ['user_id', 'status'],
        postgresql_where=sa.text('deleted_at IS NULL')
    )


def downgrade() -> None:
    """Drop documents table and related objects."""
    
    if not table_exists('documents'):
        return
    
    # Drop indexes
    op.drop_index('ix_documents_user_status', table_name='documents')
    op.drop_index('ix_documents_deleted_at', table_name='documents')
    op.drop_index('ix_documents_created_at', table_name='documents')
    op.drop_index('ix_documents_file_type', table_name='documents')
    op.drop_index('ix_documents_status', table_name='documents')
    op.drop_index('ix_documents_user_id', table_name='documents')
    
    # Drop table
    op.drop_table('documents')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS documentstatus')
    op.execute('DROP TYPE IF EXISTS documenttype')
