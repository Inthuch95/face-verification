# face-verification
Use VGGFace to calculate a face embedding for a new given face and comparing the embedding to the embedding for the single example of the face known to the system.

# Installation
Install MiniConda - https://docs.conda.io/en/latest/miniconda.html <br>
conda create --name vggface python=3.6 <br>
conda activate vggface <br>
pip install -r requirements.txt

# Usage
python main.py -i [reference image] -c [candidate image] <br>
python main.py -i data/steven_gerrard_1.jpg -c data/steven_gerrard_2.jpg
