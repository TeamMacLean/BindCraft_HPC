#!/bin/bash
#############################################
# BindCraft installer (mirror-safe, mamba-compatible)
#############################################

pkg_manager="mamba"
cuda=""
SECONDS=0

# -------------------------
# Argument parsing
# -------------------------
OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2 ;;
    -c|--cuda)
      cuda="$2"
      shift 2 ;;
    --)
      shift
      break ;;
    *)
      echo "Invalid option $1" >&2
      exit 1 ;;
  esac
done

echo "Package manager: $pkg_manager"
echo "CUDA: $cuda"

#############################################
# Enforce mirror-only behaviour
#############################################

echo "Enforcing mirror-only conda/mamba behaviour"

# Blocks fallback to Anaconda or other channel_alias values
export CONDA_CHANNEL_ALIAS="https://repo.prefix.dev"
export CONDA_OVERRIDE_CHANNELS=1
export MAMBA_NO_SSL_VERIFY=0

#############################################
# Locate conda/mamba base
#############################################

CONDA_BASE=$($pkg_manager info --base 2>/dev/null) \
  || { echo "ERROR: Could not get base for $pkg_manager"; exit 1; }

echo "Detected base at: $CONDA_BASE"

install_dir=$(pwd)

#############################################
# Create environment
#############################################

echo "Creating BindCraft environment"
$pkg_manager create -y -n BindCraft python=3.10 \
  || { echo "Error: failed to create environment"; exit 1; }

source activate BindCraft \
  || { echo "Error: could not activate BindCraft environment"; exit 1; }

echo "Environment activated."

#############################################
# Install conda packages (mirror-safe)
#############################################

echo "Installing conda packages (using mirror-only config)"

COMMON_PKGS="
pip pandas matplotlib numpy<2.0 biopython scipy pdbfixer seaborn
libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol
chex dm-haiku flax<0.10 dm-tree joblib ml-collections immutabledict optax
jax>=0.4,<=0.6.0
"

if [ -n "$cuda" ]; then
  CONDA_OVERRIDE_CUDA="$cuda" \
  $pkg_manager install -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu -y \
    $COMMON_PKGS \
    'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
    || { echo "Error: failed to install CUDA-specific packages"; exit 1; }
else
  $pkg_manager install -y \
    $COMMON_PKGS \
    jaxlib>=0.4,<=0.6.0 \
    || { echo "Error: failed to install CPU-only packages"; exit 1; }
fi

#############################################
# Validate packages
#############################################

required=(
  pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn
  tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib
  ml-collections immutabledict optax jaxlib jax
)

[ -n "$cuda" ] && required+=(cuda-nvcc cudnn)

missing=()

for pkg in "${required[@]}"; do
  conda list "$pkg" | grep -q "$pkg" || missing+=("$pkg")
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: Missing packages:"
  printf ' - %s\n' "${missing[@]}"
  exit 1
fi

#############################################
# Install ColabDesign
#############################################

echo "Installing ColabDesign"
pip3 install --no-deps "git+https://github.com/sokrypton/ColabDesign.git" \
  || { echo "Error installing ColabDesign"; exit 1; }

python -c "import colabdesign" \
  || { echo "ColabDesign import failed"; exit 1; }

#############################################
# Download AF2 model weights
#############################################

params_dir="$install_dir/params"
params_file="$params_dir/alphafold_params_2022-12-06.tar"

echo "Downloading AlphaFold2 weights..."
mkdir -p "$params_dir"

wget -O "$params_file" \
  "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
  || { echo "Weights download failed"; exit 1; }

tar -xf "$params_file" -C "$params_dir"

[ -f "$params_dir/params_model_5_ptm.npz" ] \
  || { echo "AF2 weights did not extract properly"; exit 1; }

rm "$params_file"

#############################################
# Permissions
#############################################

chmod +x "$install_dir/functions/dssp" || true
chmod +x "$install_dir/functions/DAlphaBall.gcc" || true

#############################################
# Cleanup
#############################################

source deactivate
$pkg_manager clean -a -y

t=$SECONDS
echo "BindCraft installation completed."
echo "Activate with: conda activate BindCraft"
echo "Time: $(($t / 3600))h $((($t / 60) % 60))m $(($t % 60))s"

