
def estimate_mfu(tokens_per_second: float, params: int, flops_per_token: int, peak_flops: float) -> float:
    if peak_flops <= 0:
        return 0.0
    return tokens_per_second * params * flops_per_token / peak_flops

