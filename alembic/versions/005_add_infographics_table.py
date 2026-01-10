"""Add infographics table

Revision ID: 005
Revises: 004
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create infographics table for storing generated infographic metadata."""
    
    # Create enum types
    op.execute("CREATE TYPE infographic_style AS ENUM ('modern', 'corporate', 'minimal', 'vibrant', 'dark')")
    op.execute("CREATE TYPE infographic_status AS ENUM ('pending', 'generating', 'completed', 'failed')")
    
    op.create_table('infographics',
        # Primary key
        sa.Column('id', sa.String(36), nullable=False),
        
        # User reference
        sa.Column('user_id', sa.String(36), nullable=True, comment='User who generated the infographic'),
        
        # Request details
        sa.Column('request', sa.Text(), nullable=False, comment='Original user request for the infographic'),
        sa.Column('topic', sa.String(500), nullable=True, comment='Topic used for RAG search'),
        
        # Style and dimensions
        sa.Column('style', postgresql.ENUM('modern', 'corporate', 'minimal', 'vibrant', 'dark', name='infographic_style', create_type=False), nullable=False, server_default='modern', comment='Visual style of the infographic'),
        sa.Column('width', sa.Integer(), nullable=False, server_default='1024', comment='Image width in pixels'),
        sa.Column('height', sa.Integer(), nullable=False, server_default='1024', comment='Image height in pixels'),
        
        # Structured content
        sa.Column('headline', sa.String(200), nullable=True, comment='Main headline of the infographic'),
        sa.Column('subtitle', sa.String(500), nullable=True, comment='Subtitle or context line'),
        sa.Column('structured_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}', comment='Full structured data (stats, key_points, etc.)'),
        
        # S3 storage
        sa.Column('s3_key', sa.String(500), nullable=True, comment='S3 object key for the image'),
        sa.Column('s3_bucket', sa.String(100), nullable=True, comment='S3 bucket name'),
        sa.Column('image_url', sa.String(1000), nullable=True, comment='Direct URL to the image (may be presigned)'),
        sa.Column('image_format', sa.String(10), nullable=False, server_default='png', comment='Image format (png, jpg, etc.)'),
        sa.Column('image_size_bytes', sa.Integer(), nullable=True, comment='Image file size in bytes'),
        
        # Sources and confidence
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]', comment='RAG sources used (title, date, score)'),
        sa.Column('chunks_used', sa.Integer(), nullable=False, server_default='0', comment='Number of RAG chunks used'),
        sa.Column('confidence_score', sa.Float(), nullable=True, comment='RAG confidence score (0-1)'),
        sa.Column('confidence_level', sa.String(20), nullable=True, comment='Confidence level: high, medium, low'),
        
        # Timing
        sa.Column('retrieval_ms', sa.Float(), nullable=True, comment='Time spent on RAG retrieval'),
        sa.Column('extraction_ms', sa.Float(), nullable=True, comment='Time spent on LLM data extraction'),
        sa.Column('image_gen_ms', sa.Float(), nullable=True, comment='Time spent on image generation'),
        sa.Column('total_ms', sa.Float(), nullable=True, comment='Total generation time'),
        
        # Status
        sa.Column('status', postgresql.ENUM('pending', 'generating', 'completed', 'failed', name='infographic_status', create_type=False), nullable=False, server_default='pending', comment='Generation status'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if generation failed'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True, comment='Soft delete timestamp'),
        
        # Constraints
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    )
    
    # Create indexes
    op.create_index('ix_infographics_user_id', 'infographics', ['user_id'])
    op.create_index('ix_infographics_s3_key', 'infographics', ['s3_key'])
    op.create_index('ix_infographics_status', 'infographics', ['status'])
    op.create_index('ix_infographics_user_created', 'infographics', ['user_id', 'created_at'])
    op.create_index('ix_infographics_status_created', 'infographics', ['status', 'created_at'])


def downgrade() -> None:
    """Drop infographics table."""
    op.drop_index('ix_infographics_status_created', table_name='infographics')
    op.drop_index('ix_infographics_user_created', table_name='infographics')
    op.drop_index('ix_infographics_status', table_name='infographics')
    op.drop_index('ix_infographics_s3_key', table_name='infographics')
    op.drop_index('ix_infographics_user_id', table_name='infographics')
    op.drop_table('infographics')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS infographic_status")
    op.execute("DROP TYPE IF EXISTS infographic_style")
