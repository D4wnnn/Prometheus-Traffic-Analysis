import os
import sys
import time
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("..")
from models import EncryptedTrafficClassifier
from models.light_model import LightTrafficClassifier
from finetune.finetune import TrafficDataset, collate_fn, set_seed, load_config_from_folder


def set_perf():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def autocast_ctx(amp: str):
    if amp == "none":
        return torch.cuda.amp.autocast(enabled=False)
    if amp == "fp16":
        return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    if amp == "bf16":
        return torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
    raise ValueError(f"Unknown amp={amp}")


def cm_from_preds(labels: torch.Tensor, preds: torch.Tensor, C: int) -> torch.Tensor:
    labels = labels.to(torch.int64)
    preds = preds.to(torch.int64)
    idx = labels * C + preds
    return torch.bincount(idx, minlength=C * C)


def cm_metrics(cm: np.ndarray) -> Tuple[float, float, float, float]:
    cm = cm.astype(np.float64, copy=False)
    total = cm.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    acc = float(tp.sum() / total) * 100.0
    return acc, float(prec.mean()) * 100.0, float(rec.mean()) * 100.0, float(f1.mean()) * 100.0


@dataclass
class SelectionPlan:
    strategy: str
    pkt_indices: Optional[torch.Tensor] = None         # [P]
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    byte_indices: Optional[torch.Tensor] = None        # [L]
    perpkt_byte_indices: Optional[torch.Tensor] = None # [P,L]

    @staticmethod
    def load(npz_path: Optional[str]) -> Optional["SelectionPlan"]:
        if not npz_path:
            return None
        data = np.load(npz_path, allow_pickle=True)
        cfg = json.loads(str(data["config"]))
        st = cfg.get("strategy", "contiguous")

        if st == "contiguous":
            pkt = torch.tensor(cfg["packet_indices"], dtype=torch.long)
            bs = int(cfg["byte_range"]["start"])
            be = int(cfg["byte_range"]["end"])
            return SelectionPlan(strategy=st, pkt_indices=pkt, byte_start=bs, byte_end=be)

        if st == "greedy":
            pkt = torch.tensor(cfg["packet_indices"], dtype=torch.long)
            idx_list = []
            for w in cfg["byte_windows"]:
                idx_list.append(torch.arange(int(w["start"]), int(w["end"]), dtype=torch.long))
            bidx = torch.cat(idx_list, dim=0) if len(idx_list) else torch.empty((0,), dtype=torch.long)
            return SelectionPlan(strategy=st, pkt_indices=pkt, byte_indices=bidx)

        if st == "per_packet":
            pkt_list = []
            idx_list = []
            L = None
            for pcfg in cfg["packet_configs"]:
                p = int(pcfg["packet_idx"])
                ws = pcfg["windows"]
                chunks = [torch.arange(int(w["start"]), int(w["end"]), dtype=torch.long) for w in ws]
                idx = torch.cat(chunks, dim=0) if len(chunks) else torch.empty((0,), dtype=torch.long)
                if L is None:
                    L = idx.numel()
                else:
                    if idx.numel() != L:
                        raise ValueError("per_packet windows must have same total length for gather")
                pkt_list.append(p)
                idx_list.append(idx.unsqueeze(0))  # [1,L]
            pkt = torch.tensor(pkt_list, dtype=torch.long)
            perpkt = torch.cat(idx_list, dim=0) if len(idx_list) else torch.empty((0, 0), dtype=torch.long)
            return SelectionPlan(strategy=st, pkt_indices=pkt, perpkt_byte_indices=perpkt)

        print(f"[Selector] Unknown strategy: {st}, disable selection.")
        return None

    def apply_cpu(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self is None:
            return batch
        out = {}

        bytes_x = batch.get("bytes_nlp", None)
        seq_x = batch.get("seq_nlp", None)
        tcp_x = batch.get("tcp_data", None)

        if self.strategy == "contiguous":
            pkt = self.pkt_indices
            bs, be = self.byte_start, self.byte_end
            if bytes_x is not None:
                out["bytes_nlp"] = bytes_x.index_select(1, pkt)[:, :, bs:be].contiguous()
            if seq_x is not None:
                out["seq_nlp"] = seq_x.index_select(1, pkt).contiguous()
            if tcp_x is not None:
                out["tcp_data"] = tcp_x.index_select(1, pkt).contiguous()

        elif self.strategy == "greedy":
            pkt = self.pkt_indices
            bidx = self.byte_indices
            if bytes_x is not None:
                subset = bytes_x.index_select(1, pkt)
                out["bytes_nlp"] = subset.index_select(2, bidx).contiguous()
            if seq_x is not None:
                out["seq_nlp"] = seq_x.index_select(1, pkt).contiguous()
            if tcp_x is not None:
                out["tcp_data"] = tcp_x.index_select(1, pkt).contiguous()

        elif self.strategy == "per_packet":
            pkt = self.pkt_indices
            perpkt = self.perpkt_byte_indices  # [P,L]
            if bytes_x is not None:
                subset = bytes_x.index_select(1, pkt)  # [B,P,AllB]
                idx = perpkt.unsqueeze(0).expand(subset.size(0), -1, -1)  # [B,P,L]
                out["bytes_nlp"] = torch.gather(subset, dim=2, index=idx).contiguous()
            if seq_x is not None:
                out["seq_nlp"] = seq_x.index_select(1, pkt).contiguous()
            if tcp_x is not None:
                out["tcp_data"] = tcp_x.index_select(1, pkt).contiguous()
        else:
            return batch

        for k, v in batch.items():
            if k in out:
                continue
            if k in ("bytes_nlp", "seq_nlp", "tcp_data"):
                continue
            out[k] = v
        return out


def main(args):
    set_seed(42)
    set_perf()

    # classes
    with open(os.path.join(args.data_dir, "label_mapping.json"), "r") as f:
        num_classes = len(json.load(f))

    # heavy config / dataset constraints
    model_config, heavy_max_bytes, heavy_max_packets = load_config_from_folder(args.pretrain_checkpoint_path)

    # dataset (full data)
    test_h5 = os.path.join(args.data_dir, "test_data.h5")
    ds = TrafficDataset(
        test_h5,
        augmentation=False,
        max_packets=heavy_max_packets,
        max_bytes=heavy_max_bytes,
        global_selection_path=None,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    plan = SelectionPlan.load(args.global_selector)

    dev_light = torch.device(f"cuda:{args.gpu_light}")
    dev_heavy = torch.device(f"cuda:{args.gpu_heavy}")

    # light model
    light = LightTrafficClassifier(num_classes=num_classes, d_model=64).to(dev_light)
    ckpt = torch.load(args.light_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    light.load_state_dict(state, strict=True)
    light.eval()

    # heavy model
    heavy = EncryptedTrafficClassifier(num_classes=num_classes, **model_config).to(dev_heavy)
    ckpt = torch.load(args.heavy_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    heavy.load_state_dict(state, strict=True)
    heavy.eval()

    if args.torch_compile:
        light = torch.compile(light, mode="max-autotune", fullgraph=False)
        heavy = torch.compile(heavy, mode="max-autotune", fullgraph=False)

    log_th = math.log(max(min(args.conf_threshold, 1.0 - 1e-12), 1e-12))

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # oracle buffer (CPU)
    hard_buf_x: List[Dict[str, torch.Tensor]] = []
    hard_buf_y: List[torch.Tensor] = []
    hard_count = 0
    total = 0
    heavy_calls = 0

    def flush_oracle():
        nonlocal hard_buf_x, hard_buf_y, hard_count, cm
        if hard_count <= 0:
            return
        keys = hard_buf_x[0].keys()
        x_cpu = {k: torch.cat([b[k] for b in hard_buf_x], dim=0) for k in keys}
        y_cpu = torch.cat(hard_buf_y, dim=0)

        x_gpu = {k: v.to(dev_heavy, non_blocking=True) for k, v in x_cpu.items()}
        y_gpu = y_cpu.to(dev_heavy, non_blocking=True)

        with torch.inference_mode(), autocast_ctx(args.amp):
            logits = heavy(x_gpu)
            preds = logits.argmax(dim=1)
            cm_flat = cm_from_preds(y_gpu, preds, num_classes).cpu().numpy()
        cm += cm_flat.reshape(num_classes, num_classes)

        hard_buf_x, hard_buf_y, hard_count = [], [], 0

    t0 = time.time()
    with torch.inference_mode():
        for batch in dl:
            y_cpu = batch["label"]
            x_cpu = {k: v for k, v in batch.items() if k != "label"}
            total += int(y_cpu.size(0))

            # light selection on CPU
            x_light_cpu = plan.apply_cpu(x_cpu) if plan is not None else x_cpu

            # move to light GPU
            x_light_gpu = {k: v.to(dev_light, non_blocking=True) for k, v in x_light_cpu.items()}
            y_light_gpu = y_cpu.to(dev_light, non_blocking=True)

            with autocast_ctx(args.amp):
                logits = light(x_light_gpu)
                maxlogit, preds = logits.max(dim=1)
                log_pmax = maxlogit - torch.logsumexp(logits, dim=1)
                low_conf = log_pmax < log_th

            # high-conf CM on light GPU
            high_conf = ~low_conf
            if high_conf.any():
                cm_flat = cm_from_preds(y_light_gpu[high_conf], preds[high_conf], num_classes).cpu().numpy()
                cm += cm_flat.reshape(num_classes, num_classes)

            # low-conf push to oracle buffer (slice from FULL CPU batch)
            if low_conf.any():
                m = low_conf.cpu()
                bx = {k: v[m].contiguous() for k, v in x_cpu.items()}
                by = y_cpu[m].contiguous()
                hard_buf_x.append(bx)
                hard_buf_y.append(by)
                hard_count += int(by.size(0))
                heavy_calls += int(by.size(0))

                if hard_count >= args.oracle_batch_size:
                    flush_oracle()

    flush_oracle()
    t1 = time.time()

    acc, prec, rec, f1 = cm_metrics(cm)
    dur = t1 - t0
    thr = total / dur if dur > 0 else 0.0
    heavy_rate = 100.0 * heavy_calls / total if total > 0 else 0.0

    print("\n================ ULTRAFAST SYSTEM (Single-Proc) ================")
    print(f"Total time:        {dur:.2f}s")
    print(f"Total samples:     {total}")
    print(f"Throughput:        {thr:.2f} samples/s")
    print(f"Heavy calls:       {heavy_calls} ({heavy_rate:.2f}%)")
    print("---------------------------------------------------------------")
    print(f"Accuracy:          {acc:.2f}%")
    print(f"Macro Precision:   {prec:.2f}%")
    print(f"Macro Recall:      {rec:.2f}%")
    print(f"Macro F1:          {f1:.2f}%")
    print("===============================================================\n")

    if args.save_result:
        out = {
            "mode": "System SingleProc Ultrafast",
            "config": vars(args),
            "metrics": {
                "total_time": float(dur),
                "total_samples": int(total),
                "throughput": float(thr),
                "heavy_calls": int(heavy_calls),
                "heavy_call_rate": float(heavy_rate),
                "accuracy": float(acc),
                "precision_macro": float(prec),
                "recall_macro": float(rec),
                "f1_macro": float(f1),
            },
            "confusion_matrix": cm.tolist(),
        }
        with open(args.save_result, "w") as f:
            json.dump(out, f, indent=2)
        print(f"✓ saved: {args.save_result}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--light_path", type=str, required=True)
    p.add_argument("--heavy_path", type=str, required=True)
    p.add_argument("--pretrain_checkpoint_path", type=str, required=True)
    p.add_argument("--global_selector", type=str, required=True)

    p.add_argument("--conf_threshold", type=float, default=0.75)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--gpu_light", type=int, default=0)
    p.add_argument("--gpu_heavy", type=int, default=1)

    p.add_argument("--oracle_batch_size", type=int, default=512)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--torch_compile", type=int, default=0)

    p.add_argument("--save_result", type=str, default=None)
    args = p.parse_args()
    args.pin_memory = bool(args.pin_memory)
    args.torch_compile = bool(args.torch_compile)

    main(args)
