"""entrenamiento nocturno: estimación de duración, orden de cola, actividad y preguntas nocturnas

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op

import app.db

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_jobs") as b:
        b.add_column(sa.Column("estimated_seconds", sa.Float(), nullable=True))
        b.add_column(sa.Column("queue_order", sa.Float(), nullable=False, server_default="0"))
    with op.batch_alter_table("projects") as b:
        b.add_column(sa.Column("last_activity_at", app.db.UTCDateTime(timezone=True), nullable=True))
        b.add_column(sa.Column("last_activity_by_id", sa.Integer(), nullable=True))
        b.create_foreign_key(
            "fk_projects_last_activity_by", "users", ["last_activity_by_id"], ["id"], ondelete="SET NULL"
        )
    op.create_table(
        "night_checks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("night", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False, server_default="asking"),
        sa.Column("asked_user_ids", sa.JSON(), nullable=False),
        sa.Column("answered_by_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reason", sa.String(length=200), nullable=True),
        sa.Column("created_at", app.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("answered_at", app.db.UTCDateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("project_id", "night", name="uq_night_check_project_night"),
    )
    op.create_index("ix_night_checks_project_id", "night_checks", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_night_checks_project_id", table_name="night_checks")
    op.drop_table("night_checks")
    with op.batch_alter_table("projects") as b:
        b.drop_constraint("fk_projects_last_activity_by", type_="foreignkey")
        b.drop_column("last_activity_by_id")
        b.drop_column("last_activity_at")
    with op.batch_alter_table("training_jobs") as b:
        b.drop_column("queue_order")
        b.drop_column("estimated_seconds")
