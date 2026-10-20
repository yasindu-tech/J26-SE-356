"""add subject_label to gait_analyses

Revision ID: b2f6d8e41c57
Revises: 7c4e9f12ab36
Create Date: 2026-10-20 16:45:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2f6d8e41c57"
down_revision: str | Sequence[str] | None = "7c4e9f12ab36"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: existing analyses have no label, and a label is optional.
    op.add_column("gait_analyses", sa.Column("subject_label", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("gait_analyses", "subject_label")
