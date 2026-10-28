"""importación de datasets desde Hugging Face (cuenta de cada usuario e historial de importaciones)

Revision ID: 0007
Revises: 0006
"""
import sqlalchemy as sa
from alembic import op

import app.db

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hf_connections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_enc", sa.Text(), nullable=False),
        sa.Column("token_hint", sa.String(length=16), nullable=False),
        sa.Column("hf_username", sa.String(length=100), nullable=False),
        sa.Column("created_at", app.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", app.db.UTCDateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_hf_connections_user_id", "hf_connections", ["user_id"], unique=True)

    op.create_table(
        "hf_imports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("repo_id", sa.String(length=200), nullable=False),
        sa.Column("revision", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False, server_default="pending"),
        sa.Column("phase", sa.String(length=200), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("imported", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duplicates", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("unlabeled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("rejected_sample", sa.JSON(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", app.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("started_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("finished_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", app.db.UTCDateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_hf_imports_project_id", "hf_imports", ["project_id"])
    op.create_index("ix_hf_imports_user_id", "hf_imports", ["user_id"])
    op.create_index("ix_hf_imports_status", "hf_imports", ["status"])
    op.create_index("ix_hf_imports_created_at", "hf_imports", ["created_at"])


def downgrade() -> None:
    for ix in ("created_at", "status", "user_id", "project_id"):
        op.drop_index(f"ix_hf_imports_{ix}", table_name="hf_imports")
    op.drop_table("hf_imports")
    op.drop_index("ix_hf_connections_user_id", table_name="hf_connections")
    op.drop_table("hf_connections")
