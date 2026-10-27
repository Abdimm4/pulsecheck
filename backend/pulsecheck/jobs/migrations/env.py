"""Alembic environment: migrates the database named by $DATABASE_URL."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context

from pulsecheck.jobs.models import Base
from pulsecheck.jobs.settings import DEFAULT_DATABASE_URL
from pulsecheck.jobs.store import make_engine

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    return (
        config.attributes.get("database_url")
        or os.environ.get("DATABASE_URL")
        or DEFAULT_DATABASE_URL
    )


def run_migrations_offline() -> None:
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
