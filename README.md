# FGSRA
First, install the required libraries through environment.yaml
```bash
conda env create -f environment.yaml
```
Next, specify the attack method in configs/template. yaml
Finally, use the following command to launch an attack
```bash
python main.py --config configs/template.yaml
```
use the following command to verify the attack
```bash
python verify.py --config configs/template.yaml
```
We use the dataset and models from [SSA](https://github.com/yuyang-long/SSA)