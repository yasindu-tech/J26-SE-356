"""create tapping_analyses

Revision ID: d4a7c1e95b28
Revises: b2f6d8e41c57
Create Date: 2026-10-22 09:30:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4a7c1e95b28"
down_revision: str | Sequence[str] | None = "b2f6d8e41c57"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tapping_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("clinician_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("target", sa.String(length=3), nullable=False),
        sa.Column("subject_label", sa.String(length=32), nullable=True),
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
        op.f("ix_tapping_analyses_clinician_id"), "tapping_analyses", ["clinician_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_tapping_analyses_clinician_id"), table_name="tapping_analyses")
    op.drop_table("tapping_analyses")
