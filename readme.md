conda env remove --name DM

conda create -n  DM python==3.9

conda activate DM 

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
(python版本不能太高,否则pytorch会安装失败)

pip install pandas(conda安装的版本有问题)
pip install tqdm