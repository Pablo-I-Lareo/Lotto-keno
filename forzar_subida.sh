#!/bin/bash

REPO="git@github.com:Pablo-I-Lareo/Lotto-keno.git"

echo "🧹 Borrando historial Git previo..."
rm -rf .git

echo "🔧 Inicializando nuevo repo..."
git init
git remote add origin "$REPO"

echo "📦 Añadiendo todos los archivos, incluyendo README.md correcto..."
git add -f .

echo "📝 Commit general..."
git commit -m '🔥 Subida completa del proyecto con README actualizado'

echo "📤 Push forzado a main..."
git branch -M main
git push -f origin main

echo "✅ Hecho. Revisa GitHub para confirmar que el README es el correcto."
