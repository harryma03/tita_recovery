
## 1. Training

### 1.1 Start Training
You can start the training process with either of the following commands:

```bash
# Headless training (without visualization)
python train.py --task=wheeled_titatit_recovery --headless

# Or with visualization
python train.py --task=wheeled_titatit_recovery
```

## 2. Policy-Exporting

### 2.2 Excute Policy Testing
```bash
# without visualization
python simple_play.py --task=wheeled_titatit_recovery

# without visualization
python simple_play.py --task=wheeled_titatit_recovery --headless
```