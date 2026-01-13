## TTC = Triple Tune Classifier

This is our model, which can classify the haze type and also dehaze the model.

![Proposed Model 1-01](https://github.com/user-attachments/assets/002cec73-0cc5-4542-a789-579e4ff2fa7e)

## Inference

Run inference through `RunInference.py`, which exposes subcommands for single-image, batch, and multi-image classification flows.

### Single-image TTCDehazeNet

```bash
python RunInference.py single \
  --version 2 \
  --hazy-image /path/to/hazy.jpg \
  --gt-image /path/to/gt.jpg \
  --dehazers LD_Net_Cloud,LD_Net_EH,LD_Net_Fog
```

If you are using version 1, you can optionally pass the classifier backbones:

```bash
python RunInference.py single \
  --version 1 \
  --hazy-image /path/to/hazy.jpg \
  --models DenseNet201,ResNet152,ConvNextLarge
```

### Batch dehaze + evaluate

```bash
python RunInference.py batch \
  --dehazer AllDehazer_LD_40_16_le-4_eph_35 \
  --gt-folder /path/to/gt \
  --hazy-folder /path/to/hazy
```

### Multi-image haze classification

```bash
python RunInference.py multi \
  --test-path /path/to/folder \
  --models DenseNet201,ResNet152,ConvNextLarge
```

To evaluate a single classifier backbone on the full dataset, pass just one model name:

```bash
python RunInference.py multi \
  --test-path /path/to/folder \
  --models ResNet152
```
