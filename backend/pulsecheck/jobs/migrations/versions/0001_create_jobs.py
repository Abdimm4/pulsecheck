"""Create the jobs table.

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from pulsecheck.jobs.models import UTCDateTime

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("client_hash", sa.String(64), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("upload_started_at", UTCDateTime(), nullable=True),
        sa.Column("queued_at", UTCDateTime(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("upload_expires_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.Column("upload_key", sa.String(128), nullable=True),
        sa.Column("upload_size_bytes", sa.Integer(), nullable=True),
        sa.Column("container", sa.String(8), nullable=True),
        sa.Column("deletion_status", sa.String(32), nullable=False),
        sa.Column("upload_deleted_at", UTCDateTime(), nullable=True),
        sa.Column("expired_at", UTCDateTime(), nullable=True),
        sa.Column("expiry_reason", sa.String(32), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("model_name", sa.String(64), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("report", sa.JSON(), nullable=True),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_client_hash", "jobs", ["client_hash"])
    op.create_index("ix_jobs_expires_at", "jobs", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_jobs_expires_at", "jobs")
    op.drop_index("ix_jobs_client_hash", "jobs")
    op.drop_index("ix_jobs_status", "jobs")
    op.drop_table("jobs")
