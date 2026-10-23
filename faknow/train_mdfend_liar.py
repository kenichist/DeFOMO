# train_mdfend_liar.py
# FaKnow text-only training on LIAR (binary), MDFEND on GPU.
# Needs: train.json / valid.json / test.json with {"text": "...", "label": 0/1, "domain": 0}

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from torch.utils.data import DataLoader
from faknow.data.dataset.text import TextDataset
from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.model.content_based.mdfend import MDFEND
from faknow.evaluate.evaluator import Evaluator
from faknow.train.trainer import BaseTrainer
from collections import Counter

# ---- Hyperparams ----
MAX_LEN = 170
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 5e-5
EPOCHS = 3
BERT_NAME = "bert-base-uncased"
DOMAIN_NUM = 1  # domain was set to 0 in conversion
# ---------------------

assert torch.cuda.is_available(), "GPU not visible to PyTorch"
device = "cuda"
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))

# Tokenizer
tok = TokenizerFromPreTrained(MAX_LEN, BERT_NAME)

# Datasets
train_set = TextDataset("train.json", ["text"], tok)
valid_set = TextDataset("valid.json", ["text"], tok)
test_set  = TextDataset("test.json",  ["text"], tok)

# Loaders
dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True,  **dl_kwargs)
valid_loader = DataLoader(valid_set, shuffle=False, **dl_kwargs)
test_loader  = DataLoader(test_set,  shuffle=False, **dl_kwargs)

# Model + optim
model = MDFEND(BERT_NAME, DOMAIN_NUM).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Trainer (keep version-agnostic)
trainer = BaseTrainer(model, Evaluator(), opt, device=device)

# Train
trainer.fit(train_loader, num_epochs=EPOCHS)

# Validate & test (trainer handles device for us)
val_metrics  = trainer.evaluate(valid_loader)
test_metrics = trainer.evaluate(test_loader)
print("Validation metrics:", val_metrics)
print("Test metrics:",       test_metrics)

# ---- Confusion matrix on CPU (avoid device mismatch) ----
model_cpu = model.to("cpu").eval()   # move model to CPU for manual predict

preds, golds = [], []
with torch.no_grad():
    for batch in test_loader:        # batches are CPU tensors by default
        golds.extend(batch["label"].tolist())
        out = model_cpu.predict(batch)  # model & batch both on CPU now
        if isinstance(out, torch.Tensor):
            yhat = out.argmax(dim=-1).cpu().tolist()
        else:
            try:
                yhat = list(out)
            except Exception:
                yhat = out
        preds.extend(yhat)

cm = Counter()
for y, yhat in zip(golds, preds):
    if y == 1 and yhat == 1: cm["TP"] += 1
    elif y == 0 and yhat == 1: cm["FP"] += 1
    elif y == 0 and yhat == 0: cm["TN"] += 1
    else: cm["FN"] += 1
print("Confusion (test):", dict(cm))
