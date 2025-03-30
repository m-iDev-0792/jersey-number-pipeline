echo y | conda create --name SoccerNet python=3.9
start conda activate SoccerNet
pip install gdown
pip install tqdm
pip install SoccerNet
pip install pandas
pip install opencv-python
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install numpy==1.26.4
python SetupSoccerNetDataset.py
python setup.py SoccerNet
