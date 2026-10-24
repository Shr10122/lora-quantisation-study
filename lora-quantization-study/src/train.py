import json, os, sys, torch, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
from peft import LoraConfig, get_peft_model

mode, seed = sys.argv[1], int(sys.argv[2])
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 8
set_seed(seed)

name = "roberta-base"
tok = AutoTokenizer.from_pretrained(name)
ds = load_dataset("glue", "sst2").map(
    lambda b: tok(b["sentence"], truncation=True, max_length=128), batched=True)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

if mode == "lora":
    cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.1,
                     target_modules=["query", "value"], task_type="SEQ_CLS")
    model = get_peft_model(model, cfg)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

def metrics(p):
    return {"accuracy": float((np.argmax(p.predictions, -1) == p.label_ids).mean())}

args = TrainingArguments(
    output_dir=f"out/{mode}_{seed}", per_device_train_batch_size=32,
    per_device_eval_batch_size=64, num_train_epochs=1, seed=seed, fp16=True,
    learning_rate=5e-4 if mode == "lora" else 2e-5,
    eval_strategy="no", save_strategy="no", report_to="none",
    label_names=["labels"])

trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["validation"], compute_metrics=metrics,
                  data_collator=DataCollatorWithPadding(tok))
trainer.train()
res = trainer.evaluate()

os.makedirs("results", exist_ok=True)
out = {"mode": mode, "seed": seed, "rank": rank if mode == "lora" else None,
       "accuracy": res["eval_accuracy"], "trainable_params": trainable,
       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
json.dump(out, open(f"results/{mode}_{seed}_r{rank}.json", "w"))
print(out)