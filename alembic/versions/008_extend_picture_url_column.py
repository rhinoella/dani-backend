"""Extend picture_url column to TEXT for base64 avatar storage.

Revision ID: 008
Revises: 007
Create Date: 2026-01-23

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008'
down_revision = '1e9fb64462f2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Change picture_url from VARCHAR(500) to TEXT to support base64 avatars."""
    op.alter_column(
        'users',
        'picture_url',
        existing_type=sa.VARCHAR(500),
        type_=sa.Text(),
        existing_nullable=True
    )


def downgrade() -> None:
    """Revert picture_url back to VARCHAR(500)."""
    # Note: This may truncate data if avatars are stored as base64
    op.alter_column(
        'users',
        'picture_url',
        existing_type=sa.Text(),
        type_=sa.VARCHAR(500),
        existing_nullable=True
    )
