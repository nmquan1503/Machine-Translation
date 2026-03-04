set -e

echo "Cloning Mamba..."
git clone https://github.com/nmquan1503/ssm-mamba.git ssm_mamba
cd ssm_mamba
pip install . --no-build-isolation

cd ..