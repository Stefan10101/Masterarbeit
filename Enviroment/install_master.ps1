# Minimal installer - run with:  .\install_master.ps1
conda activate Master
pip install -r requirements.txt
playwright install
Write-Host "Master environment ready." -ForegroundColor Green