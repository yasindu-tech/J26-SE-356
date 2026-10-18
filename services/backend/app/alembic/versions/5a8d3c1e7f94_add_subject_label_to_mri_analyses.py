"""add subject_label to mri_analyses

Revision ID: 5a8d3c1e7f94
Revises: 9c1e4f7a2b60
Create Date: 2026-10-18 21:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5a8d3c1e7f94"
down_revision: str | Sequence[str] | None = "9c1e4f7a2b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: existing analyses have no label, and a label is optional.
    op.add_column("mri_analyses", sa.Column("subject_label", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("mri_analyses", "subject_label")
