set -e

echo "Cloning Mamba..."
git clone https://github.com/nmquan1503/Mamba.git
cd Mamba
pip install . --no-build-isolation

cd ..