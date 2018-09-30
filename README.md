# ncc
NCC-AIエンジニア自作パッケージ

## Quick glance
```
from ncc.models import conv3d

model = conv3d(input_dim=(32, 256, 256, 3), num_classes=9)
model.summary()
```

## Installation
First, clone ncc using git:

```
git clone https://github.com/NCC-AI/ncc.git
```
Then, cd to the ncc folder and run the install command:
```
cd ncc
python setup.py install
```
