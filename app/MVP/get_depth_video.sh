#!/bin/bash
VENV_DIR="vda_venv"

# bash clone_vda.sh

echo "Input video: $1"
echo "Model: $2"
echo "Starting depth estimation ..."
source "./$VENV_DIR/bin/activate"

cd Video-Depth-Anything
python run.py --input_video ../../asset/raw/$1 --output_dir ../../asset/depth --encoder $2 --save_npz
# python run.py --input_video ../../asset/masked/$1 --output_dir ../../asset/depth --encoder $2 --save_npz
# python run.py --input_video ../../asset/raw/$1 --output_dir ../../asset/depth --encoder $2
cd ..
# Deactivate the virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
fi
