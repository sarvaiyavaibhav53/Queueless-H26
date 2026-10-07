"""Add physical counters and counter services

Revision ID: c2a4b8f10e91
Revises: 7dd9ea24a3c9
Create Date: 2026-10-07 18:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2a4b8f10e91'
down_revision: Union[str, Sequence[str], None] = '7dd9ea24a3c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add counters, counter_services, and link queue_tokens."""
    op.create_table(
        'counters',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('office_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('status', sa.Enum('ACTIVE', 'UNAVAILABLE', name='counterstatus'), nullable=False),
        sa.Column('assigned_service_id', sa.Integer(), nullable=False),
        sa.Column('status_changed_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['assigned_service_id'], ['services.id']),
        sa.ForeignKeyConstraint(['office_id'], ['offices.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_counters_assigned_service_id'), 'counters', ['assigned_service_id'], unique=False)
    op.create_index(op.f('ix_counters_id'), 'counters', ['id'], unique=False)
    op.create_index(op.f('ix_counters_office_id'), 'counters', ['office_id'], unique=False)

    op.create_table(
        'counter_services',
        sa.Column('counter_id', sa.Integer(), nullable=False),
        sa.Column('service_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['counter_id'], ['counters.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['service_id'], ['services.id']),
        sa.PrimaryKeyConstraint('counter_id', 'service_id')
    )

    op.add_column('queue_tokens', sa.Column('counter_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_queue_tokens_counter_id'), 'queue_tokens', ['counter_id'], unique=False)
    op.create_foreign_key('fk_queue_tokens_counter_id_counters', 'queue_tokens', 'counters', ['counter_id'], ['id'])


def downgrade() -> None:
    """Downgrade schema: drop counter linkage and physical counter tables."""
    op.drop_constraint('fk_queue_tokens_counter_id_counters', 'queue_tokens', type_='foreignkey')
    op.drop_index(op.f('ix_queue_tokens_counter_id'), table_name='queue_tokens')
    op.drop_column('queue_tokens', 'counter_id')
    op.drop_table('counter_services')
    op.drop_index(op.f('ix_counters_office_id'), table_name='counters')
    op.drop_index(op.f('ix_counters_id'), table_name='counters')
    op.drop_index(op.f('ix_counters_assigned_service_id'), table_name='counters')
    op.drop_table('counters')
    sa.Enum(name='counterstatus').drop(op.get_bind(), checkfirst=True)
