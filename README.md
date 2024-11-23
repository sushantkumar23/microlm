```
git clone https://github.com/sushantkumar23/microlm
cd microlm
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python cached_fineweb10B.py 10
python pretrain.py
```
