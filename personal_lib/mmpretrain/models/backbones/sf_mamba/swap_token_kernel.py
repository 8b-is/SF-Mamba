
import warnings
import torch
try:
    import triton
    import triton.language as tl
    try: 
        from functools import cached_property # to make sure cached_property can be loaded for triton
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")
except:
    warnings.warn("Triton not installed, fall back to pytorch implements.")

import time


@triton.jit
def swap_kernel(
    x_ptr,
    seq_len,
    hidden_dim,
    seqlen,
    BLOCK_SIZE: tl.constexpr,
):
    # ID combining batch and chunk
    batch_chunk_id = tl.program_id(0)
    h_block_id = tl.program_id(1)
    
    # Batch size is determined dynamically
    chunk_id = batch_chunk_id % (seq_len // seqlen)
    batch_id = batch_chunk_id // (seq_len // seqlen)
    
    chunk_start = chunk_id * seqlen
    h_start = h_block_id * BLOCK_SIZE
    
    # Hidden dimension offset
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < hidden_dim
    
    # Address calculation
    base_addr = batch_id * seq_len * hidden_dim + h_offsets
    first_addr = base_addr + chunk_start * hidden_dim
    last_addr = base_addr + (chunk_start + seqlen - 1) * hidden_dim
    
    # Data processing
    first_data = tl.load(x_ptr + first_addr, mask=h_mask)
    last_data = tl.load(x_ptr + last_addr, mask=h_mask)
    
    tl.store(x_ptr + first_addr, last_data, mask=h_mask)
    tl.store(x_ptr + last_addr, first_data, mask=h_mask)

def triton_swap(x, seqlen):
    batch_size, seq_len, hidden_dim = x.shape
    num_chunks = seq_len // seqlen
    
    BLOCK_SIZE = 64
    grid = (
        batch_size * num_chunks,
        triton.cdiv(hidden_dim, BLOCK_SIZE)
    )
    
    swap_kernel[grid](
        x, seq_len, hidden_dim, seqlen,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x


def benchmark_all_methods(batch_size=2, seq_len=49*64, hidden_dim=640, seqlen=49, num_runs=300):
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_original = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    methods = {
        'Original': lambda x: original_swap(x.clone(), seqlen),
        'Memory Efficient': lambda x: triton_swap(x.clone(), seqlen),
    }
    
    print(f"Benchmarking with shape: [{batch_size}, {seq_len}, {hidden_dim}], seqlen={seqlen}")
    print("-" * 60)
    
    results = {}
    
    for name, func in methods.items():
        # Warm up
        for _ in range(20):
            _ = func(x_original)
        
        # Actual measurement
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            result = func(x_original)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        results[name] = avg_time
        
        print(f"{name:25}: {avg_time:6.4f}ms")
    
    # Speed improvement rate calculation
    baseline = results['Original']
    print("\n" + "=" * 60)
    print("Speed improvement vs Original:")
    for name, time_ms in results.items():
        if name != 'Original (3 operations)':
            speedup = baseline / time_ms
            print(f"{name:25}: {speedup:5.2f}x faster")
    
    return results

def original_swap(x, seqlen):
    batch_size, seq_len, hidden_dim = x.shape
    st_token = x[:, ::seqlen].clone()
    x[:, ::seqlen] = x[:, seqlen-1::seqlen]
    x[:, seqlen-1::seqlen] = st_token

    return x

def verify_correctness(batch_size=3, seq_len=196*15, hidden_dim=330, seqlen=49):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    original_result = original_swap(x_test.clone(), seqlen)
    memory_opt_result = triton_swap(x_test.clone(), seqlen)

    print("Correctness verification:")
    print(f"Memory Eff vs Original: {torch.allclose(memory_opt_result, original_result, atol=1e-6)}")


if __name__ == "__main__":
    # Correctness verification
    verify_correctness()
    
    # ベンチマーク実行
    benchmark_all_methods()