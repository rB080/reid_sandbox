# Usage: ./export_env.sh myenv

ENV_NAME="$1"

if [ -z "$ENV_NAME" ]; then
  echo "Usage: $0 <conda-env-name>"
  exit 1
fi

# # Initialize conda for non-interactive shell
# CONDA_BASE=$(conda info --base)
# source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the environment
conda activate "$ENV_NAME"

# Export without builds and remove prefix
conda env export --name "$ENV_NAME" --no-builds | grep -v "^prefix:" > "${ENV_NAME}.yml"

# Deactivate environment
conda deactivate

echo "Exported environment to ${ENV_NAME}.yml"
