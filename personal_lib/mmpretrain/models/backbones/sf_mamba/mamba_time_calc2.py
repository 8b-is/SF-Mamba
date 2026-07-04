# This file obtain good batch-folding ratio. It will generate a file named n_over_batch_lut_info.py
# You can copy the output into n_over_batch_lut.py to fit your environment.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import selective_scan_cuda
from einops import rearrange, repeat
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import csv

try:
    from non_causal_conv1d import non_causal_conv1d_fn
except:
    non_causal_conv1d_fn = None

try:
    from .sf_mamba import MambaVisionModule
except:
    from sf_mamba import MambaVisionModule


def per_data(
    batch_size=128,
    dim=640,
    seqlen=225,
    dstate=8,
    n=16,
):

    model = MambaVisionModule(
        dim=dim,
        d_state=dstate,
    )

    model.cuda()
    model.eval()

    input_data = torch.randn(
        batch_size,
        dim,
        round(seqlen**0.5),
        round(seqlen**0.5),
        device="cuda",
        dtype=torch.float32,
    )
    device = "cuda"
    input_data = input_data.to(memory_format=torch.channels_last)
    model = model.to(memory_format=torch.channels_last)
    input_data = rearrange(input_data, "b c h w -> b (h w) c")

    # GPUの場合はtorch.cuda.Eventを使う
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # 推論時間の計測
    n_runs = 100

    indices = torch.zeros(
        n, (input_data.shape[0] // n) * seqlen, device=device, dtype=torch.int32
    )
    indices[:, seqlen::seqlen] = 1
    indices[:, seqlen - 1 :: seqlen] = -1
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for ii in range(n_runs + 10):
                if ii == 10:
                    torch.cuda.synchronize()
                    start_event.record()
                x_ = rearrange(input_data, "(n b) l dim -> n (b l) dim", n=n)
                output = model(x_, indices=indices)
                output = model(output[0], indices=indices)
                output = rearrange(output[0], "n (b l) dim -> (n b) l dim", l=seqlen)

        # y = rearrange(y, "n dim (b l) -> (n b) dim l", l=seqlen)
    torch.cuda.synchronize()  # Ensure both events are completed
    end_event.record()
    torch.cuda.synchronize()  # Ensure both events are completed
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / n_runs
    print(f"{avg_time_ms:.4f} ms: {batch_size} {dim} {seqlen} {dstate} {n}")
    return avg_time_ms


# Define parameter ranges
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
dims = [64, 128, 256, 512, 1024, 1600]
seqlens = [25, 49, 100, 196, 576, 1296]
dstates = [1, 2, 4, 8, 16]

# batch_sizes = [128]
# dims = [320, 640, 4*128, 8*128]
# seqlens = [49, 196]
# dstates = [8]


def list_factors(n):
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:  # 重複を防ぐ
                factors.append(n // i)
    return sorted(factors)


# Prepare CSV file
def main():
    data = []
    all_data_time = []
    for i in range(4):
        data_time = []
        for batch_size in batch_sizes:
            for dim in dims:
                for seqlen in seqlens:
                    for dstate in dstates:
                        for n in list_factors(batch_size):
                            avg_time_ms = per_data(
                                batch_size=batch_size,
                                dim=dim,
                                seqlen=seqlen,
                                dstate=dstate,
                                n=n,
                            )
                            if i == 0:
                                data.append([batch_size, dim, seqlen, dstate, n])
                            data_time.append(avg_time_ms)
        all_data_time.append(data_time)
    all_data_time = np.array(all_data_time)
    all_data_time = all_data_time.mean(0)

    with open("inference_times.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["batch_size", "dim", "seqlen", "dstate", "n", "time"])
        for i in range(len(data)):
            writer.writerow(data[i] + [all_data_time[i]])

    # ['batch_size', 'dim', 'seqlen', 'dstate'] の組み合わせでtimeが最小のnを探し、そのnを使ってn/batchを求める
    # 4次元行列 (len(batch_sizes), len(dims), len(seqlens), len(dstates)) を作成

    df = pd.read_csv("inference_times.csv")

    # 4次元配列を初期化
    n_over_batch = np.zeros(
        (len(batch_sizes), len(dims), len(seqlens), len(dstates)), dtype=np.float32
    )

    # 各パラメータのインデックスを作成
    batch_size_to_idx = {v: i for i, v in enumerate(batch_sizes)}
    dim_to_idx = {v: i for i, v in enumerate(dims)}
    seqlen_to_idx = {v: i for i, v in enumerate(seqlens)}
    dstate_to_idx = {v: i for i, v in enumerate(dstates)}

    # 各組み合わせごとにtimeが最小のnを探す
    grouped = df.groupby(["batch_size", "dim", "seqlen", "dstate"])
    for (batch_size, dim, seqlen, dstate), group in grouped:
        min_idx = group["time"].idxmin()
        best_n = int(df.loc[min_idx, "n"])
        n_over_batch[
            batch_size_to_idx[batch_size],
            dim_to_idx[dim],
            seqlen_to_idx[seqlen],
            dstate_to_idx[dstate],
        ] = best_n / batch_size

    print(n_over_batch)
    # RegularGridInterpolatorの作成
    interpolator = RegularGridInterpolator(
        (batch_sizes, dims, seqlens, dstates),
        n_over_batch,
        method="nearest",
        bounds_error=False,
        fill_value=None,
    )

    # n_over_batchをlut.pyとして保存。flatにしてlistとして保存。また、batch_sizes, dims, seqlens, dstatesも保存。
    with open("n_over_batch_lut_info.py", "w") as f:
        f.write(f"n_over_batch = {n_over_batch.flatten().tolist()}\n")
        f.write(f"batch_sizes = {batch_sizes}\n")
        f.write(f"dims = {dims}\n")
        f.write(f"seqlens = {seqlens}\n")
        f.write(f"dstates = {dstates}\n")

    # 例: 任意のパラメータで最良のn/batchを取得
    def get_best_n_over_batch(batch_size, dim, seqlen, dstate):
        point = np.array([[batch_size, dim, seqlen, dstate]])
        return interpolator(point)[0]

    # テスト出力例
    print("n_over_batch shape:", n_over_batch.shape)
    print(
        "Example: batch=32, dim=256, seqlen=196, dstate=8 -> n/batch =",
        get_best_n_over_batch(32, 256, 196, 8),
    )


if __name__ == "__main__":
    main()
