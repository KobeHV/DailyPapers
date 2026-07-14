#!/bin/bash
# Daily push script for Obsidian vault -> GitHub
# Repository: https://github.com/KobeHV/DailyPapers.git

REPO_DIR="D:/DailyPapers"
BRANCH="main"

cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR"; exit 1; }

# Pull latest first to avoid conflicts
git pull origin "$BRANCH" --rebase 2>&1 || echo "Pull warning (may be first push or no upstream)"

# Stage all changes
git add -A

# Only commit if there are changes
if git diff --cached --quiet; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to push."
    exit 0
fi

# Commit with date stamp
git commit -m "Daily sync: $(date '+%Y-%m-%d %H:%M:%S')"

# Push to GitHub
git push -u origin "$BRANCH" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Push completed."
