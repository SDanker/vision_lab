"""resumen de cada noche hábil (también cuando no se entrena nada)

Revision ID: 0006
Revises: 0005
"""
import sqlalchemy as sa
from alembic import op

import app.db

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "night_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("night", sa.String(length=10), nullable=False, unique=True),
        sa.Column("capacity_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("note", sa.String(length=300), nullable=True),
        sa.Column("items", sa.JSON(), nullable=False),
        sa.Column("not_ready", sa.JSON(), nullable=False),
        sa.Column("notified", sa.JSON(), nullable=False),
        sa.Column("created_at", app.db.UTCDateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("night_reports")
