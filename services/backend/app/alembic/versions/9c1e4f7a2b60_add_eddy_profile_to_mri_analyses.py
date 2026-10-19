"""add eddy_profile to mri_analyses

Revision ID: 9c1e4f7a2b60
Revises: 3b7d2a91c4e0
Create Date: 2026-10-18 18:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9c1e4f7a2b60"
down_revision: str | Sequence[str] | None = "3b7d2a91c4e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows were all made with the default eddy settings, hence "full".
    op.add_column(
        "mri_analyses",
        sa.Column("eddy_profile", sa.String(length=10), server_default="full", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("mri_analyses", "eddy_profile")
