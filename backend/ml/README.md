# MoodLens — ML module

Training and evaluation for the emotion classifier. Decoupled from the API:
install these deps only when doing ML work.

```bash
cd backend
pip install -r requirements-ml.txt
```

All commands run from the `backend/` directory (so `app.*` imports resolve).

## 1. Inspect the dataset

```bash
python -m ml.data
```
Prints row counts + class balance per split after mapping GoEmotions → Ekman-6.

## 2. Baseline — evaluate the off-the-shelf model (no GPU, ~minutes on CPU)

Answers the research question *today* with real numbers:

```bash
python -m ml.evaluate --model bhadresh-savani/bert-base-go-emotion --label-space goemotions
```
Writes `metrics/<model>.json` + a confusion-matrix PNG.

## 3. Fine-tune your own model (needs GPU → use Google Colab)

```bash
python -m ml.train --epochs 3 --output models/ekman-bert
```
Saves a 7-class model to `models/ekman-bert/`.

## 4. Evaluate your fine-tuned model, compare to baseline

```bash
python -m ml.evaluate --model models/ekman-bert --label-space ekman
```

## 5. Serve your fine-tuned model in the API

Set `MODEL_NAME=models/ekman-bert` (or the path where you downloaded it) in
`backend/.env`. The classifier's Ekman aggregation already handles a 7-label
model unchanged.

## Colab quickstart

```python
!git clone <your-repo-url> && cd MoodLens/backend
!pip install -r requirements-ml.txt
!python -m ml.train --epochs 3 --output models/ekman-bert
# then download models/ekman-bert/ (zip it) back to your machine
```
