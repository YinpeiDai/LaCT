import math
import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
import flax.linen as nn
from flax.core import freeze, unfreeze
from typing import Callable, Optional
import functools


@jit
def silu_backprop(dy: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    Returns:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = jax.nn.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@jit
def l2_norm(x: jnp.ndarray) -> jnp.ndarray:
    """
    x: [b, l, d]
    """
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)


@jit
def zeropower_via_newtonschulz5(G: jnp.ndarray) -> jnp.ndarray:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    """
    assert len(G.shape) == 3
    X = G.astype(jnp.bfloat16)
    
    if G.shape[1] > G.shape[2]:
        X = jnp.transpose(X, (0, 2, 1))
    
    # Ensure spectral norm is at most 1
    X = X / (jnp.linalg.norm(X, axis=(1, 2), keepdims=True) + 1e-7)
    
    # Perform the NS iterations
    coefficients = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    
    for a, b, c in coefficients:
        A = X @ jnp.transpose(X, (0, 2, 1))
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.shape[1] > G.shape[2]:
        X = jnp.transpose(X, (0, 2, 1))
    
    return X


@jit
def bidirectional_lact_swiglu(
    w0: jnp.ndarray,  # [b, dh, dk]
    w1: jnp.ndarray,  # [b, dv, dh]
    w2: jnp.ndarray,  # [b, dh, dk]
    q: jnp.ndarray,   # [b, l, dk]
    k: jnp.ndarray,   # [b, l, dk]
    v: jnp.ndarray,   # [b, l, dv]
    lr0: jnp.ndarray, # [b, l, 1]
    lr1: jnp.ndarray, # [b, l, 1]
    lr2: jnp.ndarray, # [b, l, 1]
    # use_muon: bool = True,
) -> jnp.ndarray:
    """
    Bidirectional LaCT with SwiGLU fast weight function.
    """
    # Store original norms
    w0_norm = jnp.linalg.norm(w0, axis=2, keepdims=True)
    w1_norm = jnp.linalg.norm(w1, axis=2, keepdims=True)
    w2_norm = jnp.linalg.norm(w2, axis=2, keepdims=True)

    q = jnp.transpose(q, (0, 2, 1))  # [b, dk, l]
    v = jnp.transpose(v, (0, 2, 1))  # [b, dv, l]

    ######### Update the fast weight w0, w1, w2 with test-time training #########

    #### Forward pass with key
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    gate_before_act = w0 @ jnp.transpose(k, (0, 2, 1))
    hidden_before_mul = w2 @ jnp.transpose(k, (0, 2, 1))
    hidden = jax.nn.silu(gate_before_act) * hidden_before_mul

    #### Backward pass to compute fast weight gradients
    # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
    dhidden = jnp.transpose(w1, (0, 2, 1)) @ v

    dhidden_before_mul = dhidden * jax.nn.silu(gate_before_act)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    # Compute gradients
    # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
    dw1 = v @ (jnp.transpose(hidden, (0, 2, 1)) * lr1)
    # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
    dw0 = dgate_before_act @ (k * lr0)
    dw2 = dhidden_before_mul @ (k * lr2)

    w0_update = zeropower_via_newtonschulz5(dw0)
    w1_update = zeropower_via_newtonschulz5(dw1)
    w2_update = zeropower_via_newtonschulz5(dw2)
        
    # if use_muon:
    #     w0_update = zeropower_via_newtonschulz5(dw0)
    #     w1_update = zeropower_via_newtonschulz5(dw1)
    #     w2_update = zeropower_via_newtonschulz5(dw2)
    # else:
    #     w0_update = jnp.zeros_like(w0)
    #     w1_update = jnp.zeros_like(w1)
        # w2_update = jnp.zeros_like(w2)

    # Update weights
    w1 = w1 + w1_update + dw1
    w0 = w0 + w0_update + dw0
    w2 = w2 + w2_update + dw2

    # Normalize weights
    w0 = w0 / (jnp.linalg.norm(w0, axis=2, keepdims=True) + 1e-5) * w0_norm
    w1 = w1 / (jnp.linalg.norm(w1, axis=2, keepdims=True) + 1e-5) * w1_norm
    w2 = w2 / (jnp.linalg.norm(w2, axis=2, keepdims=True) + 1e-5) * w2_norm

    ######### Apply the updated fast weights to the query #########

    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = w2 @ q
    gate = jax.nn.silu(w0 @ q)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    o = jnp.transpose(w1 @ (gate * h), (0, 2, 1))

    return o


def inv_softplus(x):
    """Inverse softplus function."""
    if isinstance(x, jnp.ndarray):
        y = x + jnp.log(-jnp.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class RMSNorm(nn.Module):
    """RMS normalization layer."""
    dim: int
    eps: float = 1e-5
    
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return scale * x / norm


class BidirectionalLaCTSwiGLU(nn.Module):
    """Bidirectional LaCT SwiGLU layer."""
    
    dim: int
    head_dim: int
    inter_multi: float = 1.0
    use_o_norm: bool = True
    qk_l2_norm: bool = True
    use_muon: bool = True
    base_lr: float = 1e-2

    def setup(self):
        self.num_heads = self.dim // self.head_dim
        
        self.to_qkv = nn.Dense(3 * self.dim, use_bias=False)
        self.o_proj = nn.Dense(self.dim, use_bias=False)
        
        self.lr_dim = 1
        self.lr_proj = nn.Dense(self.lr_dim * 3 * self.num_heads, use_bias=False)
        self.base_lr_inv = inv_softplus(self.base_lr)
        
        # Create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)
        
        # Initialize parameters
        self.w0 = self.param(
            'w0', 
            lambda rng, shape: jax.random.normal(rng, shape) / math.sqrt(d_in),
            (self.num_heads, d_h, d_in)
        )
        self.w1 = self.param(
            'w1',
            lambda rng, shape: jax.random.normal(rng, shape) / math.sqrt(d_h),
            (self.num_heads, d_out, d_h)
        )
        self.w2 = self.param(
            'w2',
            lambda rng, shape: jax.random.normal(rng, shape) / math.sqrt(d_in),
            (self.num_heads, d_h, d_in)
        )
        
        if self.use_o_norm:
            self.o_norm = RMSNorm(self.head_dim)
        else:
            self.o_norm = lambda x: x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: [b, l, d]
        Returns:
            output: [b, l, d]
        """
        batch_size, seq_len, _ = x.shape
        
        qkv = jax.nn.silu(self.to_qkv(x))
        
        # Reshape to [b * num_heads, l, head_dim] for q, k, v
        qkv_reshaped = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_reshaped = jnp.transpose(qkv_reshaped, (2, 0, 3, 1, 4))  # [3, b, h, l, d]
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        
        # Reshape to [b * num_heads, l, head_dim]
        q = q.reshape(-1, seq_len, self.head_dim)
        k = k.reshape(-1, seq_len, self.head_dim)
        v = v.reshape(-1, seq_len, self.head_dim)
        
        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)
        
        # Compute learning rates
        lr = self.lr_proj(x)  # [b, l, lr_dim * 3 * num_heads]
        lr = jax.nn.softplus(lr + self.base_lr_inv)
        
        # Reshape learning rates to [3, b * num_heads, l, 1]
        lr_reshaped = lr.reshape(batch_size, seq_len, 3, self.num_heads, self.lr_dim)
        lr_reshaped = jnp.transpose(lr_reshaped, (2, 0, 3, 1, 4))  # [3, b, h, l, d]
        lr0, lr1, lr2 = lr_reshaped[0], lr_reshaped[1], lr_reshaped[2]
        
        # Reshape to [b * num_heads, l, 1]
        lr0 = lr0.reshape(-1, seq_len, self.lr_dim)
        lr1 = lr1.reshape(-1, seq_len, self.lr_dim)
        lr2 = lr2.reshape(-1, seq_len, self.lr_dim)
        
        # Expand fast weights to batch dimension
        # [num_heads, d, d] -> [b * num_heads, d, d]
        w0_expanded = jnp.tile(self.w0, (batch_size, 1, 1))
        w1_expanded = jnp.tile(self.w1, (batch_size, 1, 1))
        w2_expanded = jnp.tile(self.w2, (batch_size, 1, 1))
        
        # Apply bidirectional LaCT
        output = bidirectional_lact_swiglu(
            w0_expanded, w1_expanded, w2_expanded, 
            q, k, v, lr0, lr1, lr2
        )
        
        # Apply output normalization
        if self.use_o_norm:
            output = self.o_norm(output)
        
        # Reshape back to [b, l, d]
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        output = jnp.transpose(output, (0, 2, 1, 3))  # [b, l, h, d]
        output = output.reshape(batch_size, seq_len, self.dim)
        
        # Final projection
        output = self.o_proj(output)
        
        return output


def test_layer():
    """Test function for the BidirectionalLaCTSwiGLU layer."""
    B, L, D, HeadDim = 4, 32768, 2048, 512 # Smaller dimensions for testing
    
    # Initialize model
    model = BidirectionalLaCTSwiGLU(D, HeadDim, use_muon=True)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (B, L, D))
    
    # Initialize model parameters
    params = model.init(key, x)
    
    # JIT compile the forward pass
    @jit
    def forward(params, x):
        return model.apply(params, x)
    
    # Warmup
    for _ in range(10):
        output = forward(params, x)
    
    # Timing
    start_time = time.time()
    num_runs = 100
    for _ in range(num_runs):
        output = forward(params, x).block_until_ready()
    end_time = time.time()
    
    print(f"Time taken: {(end_time - start_time) / num_runs:.4f} seconds per sample")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Input norm: {jnp.linalg.norm(x):.4f}, Output norm: {jnp.linalg.norm(output):.4f}")


if __name__ == "__main__":
    test_layer()