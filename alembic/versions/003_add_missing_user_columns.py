"""Add missing columns to users table

Revision ID: 003
Revises: 002
Create Date: 2025-12-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c['name'] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # Add missing columns to users table (only if they don't exist)
    if not column_exists('users', 'metadata'):
        op.add_column('users', sa.Column('metadata', postgresql.JSONB(), nullable=True, server_default='{}'))
    
    if not column_exists('users', 'is_active'):
        op.add_column('users', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # Add tokens_used column to messages if it doesn't exist
    if not column_exists('messages', 'tokens_used'):
        op.add_column('messages', sa.Column('tokens_used', sa.Integer(), nullable=True))
    
    # Add missing columns to conversations if needed
    if not column_exists('conversations', 'is_archived'):
        op.add_column('conversations', sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='false'))
    
    if not column_exists('conversations', 'is_pinned'):
        op.add_column('conversations', sa.Column('is_pinned', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    # Remove added columns in reverse order (only if they exist)
    if column_exists('conversations', 'is_pinned'):
        op.drop_column('conversations', 'is_pinned')
    if column_exists('conversations', 'is_archived'):
        op.drop_column('conversations', 'is_archived')
    if column_exists('messages', 'tokens_used'):
        op.drop_column('messages', 'tokens_used')
    if column_exists('users', 'is_active'):
        op.drop_column('users', 'is_active')
    if column_exists('users', 'metadata'):
        op.drop_column('users', 'metadata')
