# Create virtual env
python3 -m venv .venv
source .venv/bin/activate  
# Install pip and ML libraries
sudo apt install python3-venv python3-pip
# Pip upgrade
python -m ensurepip --upgrade

# Ignore data push

echo "*.csv" >> .gitignore
