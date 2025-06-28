import torch

def cumulative_outer_products(A: torch.Tensor, B: torch.Tensor, l: int):
    """
    Computes cumulative outer products:
    C_i = B[:, :l*i] @ A[:l*i, :] for i in [1, ..., L//l]

    Args:
        A: (L, d)
        B: (d, L)
        l: Chunk length

    Returns:
        Tensor of shape (L//l, d, d)
    """
    L, d = A.shape
    assert B.shape == (d, L), "B must be of shape (d, L)"
    assert L % l == 0, "L must be divisible by l"

    # Precompute cumulative matmuls: prefix sum of outer products
    # B @ A = sum_{i=0}^{L-1} B[:, i] outer A[i]
    # Construct partial sums

    # (d, L) @ (L, d) => (d, d), via cumulative sum trick
    outer_products = torch.einsum('dl,lm->dml', B, A)  # (d, d, l)

    # Cumulative sum along L axis
    cum_outer = torch.cumsum(outer_products, dim=-1)  # (d, d, l)

    # Sample every l steps: [:, :, l-1], [:, :, 2l-1], ...
    num_chunks = L // l
    idxs = torch.arange(1, num_chunks + 1, device=A.device) * l - 1
    results = cum_outer[:, :, idxs]  # (d, d, num_chunks)

    # Rearrange to (num_chunks, d, d)
    results = results.permute(2, 0, 1).contiguous()
    return results  # (num_chunks, d, d)

# Example
L, d, l = 1024, 4, 4
A = torch.ones(L, d, device='cuda')
B = torch.ones(d, L, device='cuda')

result = cumulative_outer_products(A, B, l)
print(result.shape)  # should be (L//l, d, d)
print(result)
