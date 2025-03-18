
PYTHON_VERSION="python3.10"
VENV_DIR="vda_venv"


# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment ($VENV_DIR)..."
    $PYTHON_VERSION -m venv "$VENV_DIR"
else
    echo "Virtual environment ($VENV_DIR) already exists."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
# Upgrade pip
pip install --upgrade pip

# Deactivate the virtual environment once
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything

pip install -r requirements.txt

cd ..

# Deactivate the virtual environment once
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
fi

echo "... Video-Depth-Anything clone and venv initialisation complete ..."