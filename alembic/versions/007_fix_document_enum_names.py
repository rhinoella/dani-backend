"""Fix document enum names to match model

Revision ID: 007
Revises: 006
Create Date: 2026-01-10 14:00:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Skip if documents table doesn't exist
    if 'documents' not in inspector.get_table_names():
        return

    # Check if migration already applied by checking if document_type exists
    result = bind.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'document_type'"))
    if result.fetchone() is not None:
        # Migration already applied
        return

    # 1. Ensure new types exist with correct lowercase values (matching python Enum)
    # Drop and recreate to ensure clean state
    op.execute("DROP TYPE IF EXISTS document_type CASCADE")
    op.execute("DROP TYPE IF EXISTS document_status CASCADE")

    op.execute("""
        CREATE TYPE document_type AS ENUM ('pdf', 'docx', 'txt');
    """)

    op.execute("""
        CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed');
    """)

    # 2. Convert columns to use the new types
    # First drop defaults to avoid casting errors
    op.execute("ALTER TABLE documents ALTER COLUMN status DROP DEFAULT")

    # We cast via text to ensure compatibility
    op.execute("""
        ALTER TABLE documents
        ALTER COLUMN file_type TYPE document_type
        USING lower(file_type::text)::document_type
    """)

    op.execute("""
        ALTER TABLE documents
        ALTER COLUMN status TYPE document_status
        USING lower(status::text)::document_status
    """)

    # Restore default with new type
    op.execute("ALTER TABLE documents ALTER COLUMN status SET DEFAULT 'pending'::document_status")

    # 3. Drop the old types (originally created as documenttype/documentstatus)
    op.execute("DROP TYPE IF EXISTS documenttype")
    op.execute("DROP TYPE IF EXISTS documentstatus")

def downgrade() -> None:
    # Reverse operation
    op.execute("""
        CREATE TYPE documenttype AS ENUM ('pdf', 'docx', 'txt', 'markdown', 'other');
        CREATE TYPE documentstatus AS ENUM ('pending', 'processing', 'completed', 'failed', 'deleted');
    """)
    
    op.execute("""
        ALTER TABLE documents 
        ALTER COLUMN file_type TYPE documenttype 
        USING file_type::text::documenttype
    """)
    
    op.execute("""
        ALTER TABLE documents 
        ALTER COLUMN status TYPE documentstatus 
        USING status::text::documentstatus
    """)
    
    op.execute("""
        DROP TYPE document_type;
        DROP TYPE document_status;
    """)
