#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables if .env exists
if [ -f "$PARENT_DIR/.env" ]; then
  # Source the backend .env file to get SUPABASE_URL
  # Using 'set -a' allows exporting variables loaded from the file
  set -a
  source "$PARENT_DIR/.env"
  set +a
fi

# Determine command (default: info)
COMMAND=${1:-info}
shift 2>/dev/null || true

# Extract project reference from SUPABASE_URL if present
if [ -n "$SUPABASE_URL" ]; then
  PROJECT_REF=$(echo "$SUPABASE_URL" | sed -E 's|https://([^/:]+)\.supabase\.co.*|\1|')
else
  PROJECT_REF="wouemldzkrijfjgabqdb"
fi

# Connect using Supabase's Session Pooler to support IPv4 on Docker.
# Your project is assigned to aws-1-ap-northeast-2.pooler.supabase.com
DB_HOST="aws-1-ap-northeast-2.pooler.supabase.com"
DB_PORT=${DB_PORT:-5432}
DB_USER="postgres.$PROJECT_REF"
DB_NAME=${DB_NAME:-postgres}



# Prompt for password if not set in environment
if [ -z "$DB_PASSWORD" ]; then
  echo -n "Enter your Supabase Database Password: "
  read -s DB_PASSWORD
  echo ""
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

echo "Running Flyway '$COMMAND' against $DB_HOST:$DB_PORT..."

# Run Flyway using Docker with host network mode to support IPv6
docker run --rm \
  --network host \
  -v "$SCRIPT_DIR/migration:/flyway/sql" \
  flyway/flyway:10.15.0 \
  -url="jdbc:postgresql://$DB_HOST:$DB_PORT/$DB_NAME" \
  -user="$DB_USER" \
  -password="$DB_PASSWORD" \
  -connectRetries=60 \
  "$COMMAND" "$@"

