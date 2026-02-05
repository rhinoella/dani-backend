"""add_processed_at_column_to_documents

Revision ID: 847f4fe85522
Revises: c11e39ac8655
Create Date: 2026-02-05 09:31:41.829322+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '847f4fe85522'
down_revision: Union[str, None] = 'c11e39ac8655'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add processed_at column to documents table if it doesn't exist."""

    # Check if the column already exists before adding it (idempotent)
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Check if documents table exists
    if 'documents' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('documents')}

        # Add processed_at column if it doesn't exist
        if 'processed_at' not in columns:
            op.add_column('documents', sa.Column(
                'processed_at',
                sa.DateTime(timezone=True),
                nullable=True,
                comment='When processing completed'
            ))

            # If processing_completed_at exists, copy its data to processed_at
            if 'processing_completed_at' in columns:
                op.execute("""
                    UPDATE documents
                    SET processed_at = processing_completed_at
                    WHERE processing_completed_at IS NOT NULL
                """)


def downgrade() -> None:
    """Remove processed_at column from documents table."""

    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if 'documents' in inspector.get_table_names():
        columns = {col['name'] for col in inspector.get_columns('documents')}

        if 'processed_at' in columns:
            op.drop_column('documents', 'processed_at')
