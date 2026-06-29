# How to Train LegalSum on Custom Court Proceeding Videos

To train the unsupervised multimodal model on your own court videos:

## 1. Extract Features to H5 Format
Create an H5 dataset file containing the extracted GoogLeNet features, motion scores, and audio loudness profiles.

We have provided a helper feature extraction script `demo/prepare_dataset.py` that processes a folder of raw videos and outputs a single `.h5` file.

### Run Extraction:
```bash
python demo/prepare_dataset.py --video-dir /path/to/court_videos --output-h5 datasets/court_dataset.h5
```

---

## 2. Generate Split JSON
Generate train/test splits for cross-validation:
```bash
python create_split.py -d datasets/court_dataset.h5 --save-dir datasets --save-name court_splits --num-splits 5
```

---

## 3. Train Model
Run the scheduled training script. Since it is unsupervised, it optimizes policy gradients via REINFORCE:
```bash
export OMP_NUM_THREADS=$(nproc); export MKL_NUM_THREADS=$(nproc); \
python main.py \
    -d datasets/court_dataset.h5 \
    -s datasets/court_splits.json \
    -m custom \
    --save-dir log/court-model \
    --split-id 1 \
    --max-epoch 30 \
    --phase2-epochs 15 \
    --lr 1e-4 \
    --model-type enhanced \
    --hidden-dim 256 \
    --num-heads 8 \
    --num-layers 2 \
    --dropout 0.40 \
    --entropy-start 0.10 \
    --entropy-end 0.01 \
    --ensemble-k 1 \
    --use-cpu
```
