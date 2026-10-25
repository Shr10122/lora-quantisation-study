import json, os, sys, torch, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
from peft import LoraConfig, get_peft_model

mode, seed = sys.argv[1], int(sys.argv[2])
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 8
epochs = int(os.environ.get("EPOCHS", 1))
set_seed(seed)

name = "roberta-base"
tok = AutoTokenizer.from_pretrained(name)
ds = load_dataset("nyu-mll/glue", "sst2").map(
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
    per_device_eval_batch_size=64, num_train_epochs=epochs, seed=seed, fp16=True,
    learning_rate=5e-4 if mode == "lora" else 2e-5,
    eval_strategy="no", save_strategy="no", report_to="none",
    label_names=["labels"])

trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["validation"], compute_metrics=metrics,
                  data_collator=DataCollatorWithPadding(tok))
trainer.train()
res = trainer.evaluate()
print(res)

if mode == "full":
    base = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    sv = {}
    for (n, p), (_, p0) in zip(model.named_parameters(), base.named_parameters()):
        if n.endswith(("attention.self.query.weight", "attention.self.value.weight")):
            w, w0 = p.detach().float().cpu(), p0.detach().float().cpu()
            sv[n] = torch.linalg.svdvals(w - w0).numpy()
            sv[n + ".W0"] = torch.linalg.svdvals(w0).numpy()
    np.savez(f"results/dW_sv_{mode}_{seed}_e{epochs}.npz", **sv)

if mode == "full" and os.environ.get("TRUNC") == "1":
    os.makedirs("results", exist_ok=True)
    base = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    base_p = dict(base.named_parameters())
    params = dict(model.named_parameters())
    targets = [n for n in params
               if n.endswith(("attention.self.query.weight", "attention.self.value.weight"))]
    orig = {n: params[n].detach().clone() for n in targets}  # fine-tuned weights
    svd = {n: torch.linalg.svd(orig[n].float().cpu() - base_p[n].detach().float(),
                               full_matrices=False) for n in targets}
    curve = {}
    for r in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 768]:
        with torch.no_grad():
            for n in targets:
                U, S, Vh = svd[n]
                approx = (U[:, :r] * S[:r]) @ Vh[:r]
                new_w = base_p[n].detach().float() + approx
                params[n].copy_(new_w.to(params[n].device, params[n].dtype))
        curve[r] = trainer.evaluate()["eval_accuracy"]
        print("rank", r, "accuracy", round(curve[r] * 100, 2))
    with torch.no_grad():                      # restore the fine-tuned weights
        for n in targets:
            params[n].copy_(orig[n])
    json.dump(curve, open(f"results/trunc_{seed}_e{epochs}.json", "w"))
os.makedirs("results", exist_ok=True)
out = {"mode": mode, "seed": seed, "rank": rank if mode == "lora" else None,
       "epochs": epochs, "accuracy": res["eval_accuracy"],
       "trainable_params": trainable,
       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
json.dump(out, open(f"results/{mode}_{seed}_r{rank}_e{epochs}.json", "w"))
print(out)
