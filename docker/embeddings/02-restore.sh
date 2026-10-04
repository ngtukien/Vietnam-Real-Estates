#!/usr/bin/env bash
set -euo pipefail

# Các phần là những đoạn liên tiếp của cùng file gzip; ghép theo tên rồi giải nén theo luồng.
# Không cần tạo thêm một file SQL lớn trong container khi khởi tạo volume trống.
cat /opt/embeddings/parts/snapshot-*.part | gzip -dc | \
    psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --no-password --set ON_ERROR_STOP=1
