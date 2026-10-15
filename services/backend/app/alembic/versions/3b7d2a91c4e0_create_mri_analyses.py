"""create mri_analyses

Revision ID: 3b7d2a91c4e0
Revises: efc17d605653
Create Date: 2026-10-15 12:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3b7d2a91c4e0"
down_revision: str | Sequence[str] | None = "efc17d605653"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mri_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("clinician_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("age", sa.Float(), nullable=False),
        sa.Column("sex", sa.String(length=1), nullable=False),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["clinician_id"], ["clinicians.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_mri_analyses_clinician_id"), "mri_analyses", ["clinician_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_mri_analyses_clinician_id"), table_name="mri_analyses")
    op.drop_table("mri_analyses")
