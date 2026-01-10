"""Add S3 storage columns to documents table.

Revision ID: 006
Revises: 005
Create Date: 2025-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    result = bind.execute(sa.text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = :table AND column_name = :column
        )
    """), {"table": table_name, "column": column_name})
    return result.scalar()


def upgrade() -> None:
    # Add storage_key column if not exists
    if not column_exists('documents', 'storage_key'):
        op.add_column(
            'documents',
            sa.Column(
                'storage_key',
                sa.String(1000),
                nullable=True,
                comment='S3 object key for the original file'
            )
        )
    
    # Add storage_bucket column if not exists
    if not column_exists('documents', 'storage_bucket'):
        op.add_column(
            'documents',
            sa.Column(
                'storage_bucket',
                sa.String(100),
                nullable=True,
                comment='S3 bucket name'
            )
        )
    
    # Add index on storage_key for lookups
    try:
        op.create_index('ix_documents_storage_key', 'documents', ['storage_key'])
    except Exception:
        pass  # Index may already exist


def downgrade() -> None:
    # Drop index
    try:
        op.drop_index('ix_documents_storage_key', 'documents')
    except Exception:
        pass
    
    # Drop columns
    if column_exists('documents', 'storage_bucket'):
        op.drop_column('documents', 'storage_bucket')
    
    if column_exists('documents', 'storage_key'):
        op.drop_column('documents', 'storage_key')
