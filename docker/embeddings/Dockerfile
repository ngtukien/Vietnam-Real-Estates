# Image chứa dữ liệu embedding đã tính, KG và báo cáo GraphRAG.
# Đây là template; embed.py sinh build/embeddings/Dockerfile với một COPY cho mỗi phần snapshot.
FROM pgvector/pgvector:0.8.7-pg17-bookworm

LABEL org.opencontainers.image.title="Vietnam Real Estates embeddings" \
      org.opencontainers.image.description="PostgreSQL + pgvector with precomputed E5, KG and GraphRAG indexes"

# Extension phải tồn tại trước khi tạo các cột halfvec trong snapshot.
COPY 01-pgvector.sql /docker-entrypoint-initdb.d/01-pgvector.sql
# SNAPSHOT_LAYERS
COPY --chmod=0755 02-restore.sh /docker-entrypoint-initdb.d/02-restore.sh
COPY manifest.json /opt/embeddings/manifest.json
