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
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Skip if users table doesn't exist
    if 'users' not in inspector.get_table_names():
        return

    # Check if column exists and what type it is
    columns = {col['name']: col for col in inspector.get_columns('users')}
    if 'picture_url' in columns:
        # Only alter if it's not already TEXT
        col_type = str(columns['picture_url']['type'])
        if 'TEXT' not in col_type.upper():
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
