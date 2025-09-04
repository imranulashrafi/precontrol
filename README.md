# 🧠 PRE-Control

## 📦 Requirements

- Python >= 3.11  
- 2 GPUs with 24GB memory each
- 1.5 TB disk space

## 🚀 Getting Started

- Clone the repo:

```bash
git clone https://github.com/imranulashrafi/alignet.git
```

- Set up the environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

- Download dataset files from here https://tinyurl.com/shpdata and place them at `dataset/shp`.

- Specify config parameters in `experiments/config.yaml`.

- Login to W&B using the following:

```bash
wandb login
```

- Export PYTHONPATH:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

- Run training:
```bash
python scripts/trainer.py
```


## 🔬 Testing

- Download pre trained epxeriment artifacts from here: https://tinyurl.com/precontrolexperiments and place them in `experiments`.

- Specify config parameters in `experiments/config.yaml`.

- Run testing:
```bash
python scripts/test.py
```