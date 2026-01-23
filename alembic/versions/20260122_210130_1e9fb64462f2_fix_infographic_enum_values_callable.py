"""Fix infographic enum values_callable

Revision ID: 1e9fb64462f2
Revises: 007
Create Date: 2026-01-22 21:01:30.814328+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e9fb64462f2'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
