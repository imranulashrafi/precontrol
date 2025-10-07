# 🧠 Pref-CTRL

## 📦 Requirements

- Python >= 3.11  
- 2 GPUs with 24GB memory each
- 1.5 TB disk space

## 🚀 Getting Started

- Set up the environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

- Export PYTHONPATH:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

- Generate dataset files using the following:
```bash
python scripts/precompute_shp_data.py
python scripts/precompute_rewards.py
```

- Specify config parameters in `experiments/config.yaml`.

- Login to W&B using the following:

```bash
wandb login
```

- Run training:
```bash
python scripts/trainer.py
```


## 🔬 Testing
- Specify config parameters in `experiments/config.yaml`.

- Run testing:
```bash
python scripts/test.py
```