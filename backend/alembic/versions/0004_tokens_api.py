"""tokens de API personales (lectura y escritura con aprobación)

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

import app.db

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("prefix", sa.String(length=16), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=5), nullable=False, server_default="read"),
        sa.Column("created_at", app.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("expires_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("last_used_ip", sa.String(length=64), nullable=True),
        sa.Column("revoked_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("revoked_by_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_api_tokens_user_id", "api_tokens", ["user_id"])
    op.create_index("ix_api_tokens_token_hash", "api_tokens", ["token_hash"], unique=True)

    op.create_table(
        "api_token_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=10), nullable=False, server_default="pending"),
        sa.Column("expires_days", sa.Integer(), nullable=True),
        sa.Column("created_at", app.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("reviewed_by_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("token_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_api_token_requests_user_id", "api_token_requests", ["user_id"])
    op.create_index("ix_api_token_requests_status", "api_token_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_api_token_requests_status", table_name="api_token_requests")
    op.drop_index("ix_api_token_requests_user_id", table_name="api_token_requests")
    op.drop_table("api_token_requests")
    op.drop_index("ix_api_tokens_token_hash", table_name="api_tokens")
    op.drop_index("ix_api_tokens_user_id", table_name="api_tokens")
    op.drop_table("api_tokens")
