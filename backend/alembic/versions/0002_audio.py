"""proyectos de audio: tipo de proyecto y duración de archivos

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("modality", sa.String(length=10), nullable=False, server_default="image"))
    op.add_column("images", sa.Column("duration_s", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("images", "duration_s")
    op.drop_column("projects", "modality")
