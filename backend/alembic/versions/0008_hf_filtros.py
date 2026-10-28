"""importación de Hugging Face: contador de imágenes omitidas por filtros y máximos

Revision ID: 0008
Revises: 0007
"""
import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("hf_imports", sa.Column("skipped", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("hf_imports", "skipped")
