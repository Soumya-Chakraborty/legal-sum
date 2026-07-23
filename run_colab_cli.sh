#!/bin/bash
# run_colab_cli.sh — Automate GPU training using google-colab-cli

SESSION_NAME="legal-sum-gpu"

echo "============================================================"
echo "Creating or connecting to Colab session 'legal-sum-gpu'..."
echo "============================================================"
colab new --gpu T4 --session "$SESSION_NAME" || true
colab drivemount -s "$SESSION_NAME"

echo "============================================================"
echo "Executing training pipeline on remote Colab VM (with 24h timeout)..."
echo "============================================================"
colab exec --session "$SESSION_NAME" --timeout 86400 -f /home/developer/GoogleDrive/legal_sum/run_training_on_colab.py

echo "============================================================"
echo "Done! Keeping the session alive so you can check status."
echo "Use 'colab stop -s $SESSION_NAME' when done to release VM."
echo "============================================================"
