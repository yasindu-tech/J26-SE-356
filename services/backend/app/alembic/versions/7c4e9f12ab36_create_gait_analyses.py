"""create gait_analyses

Revision ID: 7c4e9f12ab36
Revises: 3b7d2a91c4e0
Create Date: 2026-10-18 16:30:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c4e9f12ab36"
down_revision: str | Sequence[str] | None = "3b7d2a91c4e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gait_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("clinician_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("one_person_confirmed", sa.Boolean(), nullable=False),
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
        op.f("ix_gait_analyses_clinician_id"), "gait_analyses", ["clinician_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_gait_analyses_clinician_id"), table_name="gait_analyses")
    op.drop_table("gait_analyses")
