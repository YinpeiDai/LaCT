import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
import math


def silu(x):
    return x * jax.nn.sigmoid(x)


def silu_backprop(dy, x):
    sigma = jax.nn.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def l2_norm(x, axis=-1, eps=1e-5):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def zeropower_via_newtonschulz5(G):
    X = G
    transposed = False
    if G.shape[1] > G.shape[2]:
        X = jnp.transpose(X, (0, 2, 1))
        transposed = True
    X = X / (jnp.linalg.norm(X, axis=(1, 2), keepdims=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ jnp.transpose(X, (0, 2, 1))
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = jnp.transpose(X, (0, 2, 1))
    return X


def inv_softplus(x):
    return x + jnp.log(-jnp.expm1(-x))


@partial(jax.jit, static_argnames=("chunk_size", "use_muon"))
def block_causal_lact_swiglu(
    w0, w1, w2, q, k, v, lr0, lr1, lr2, momentum=None, use_muon=True, chunk_size=2048
):
    b, l, dk = k.shape
    v = jnp.transpose(v, (0, 2, 1))
    q = jnp.transpose(q, (0, 2, 1))
    output = jnp.zeros_like(v)

    w0_norm = jnp.linalg.norm(w0, axis=2, keepdims=True)
    w1_norm = jnp.linalg.norm(w1, axis=2, keepdims=True)
    w2_norm = jnp.linalg.norm(w2, axis=2, keepdims=True)

    for i in range(0, l - chunk_size, chunk_size):
        s, e = i, i + chunk_size
        ki = k[:, s:e, :]
        vi = v[:, :, s:e]
        qi = q[:, :, s:e]
        lr0i, lr1i, lr2i = lr0[:, s:e, :], lr1[:, s:e, :], lr2[:, s:e, :]

        h = jnp.einsum("bij,bjk->bik", w2, qi)
        gate = silu(jnp.einsum("bij,bjk->bik", w0, qi))
        output = output.at[:, :, s:e].set(jnp.einsum("bij,bjk->bik", w1, gate * h))

        gate_before = jnp.einsum("bij,bjk->bik", w0, jnp.transpose(ki, (0, 2, 1)))
        hidden_before = jnp.einsum("bij,bjk->bik", w2, jnp.transpose(ki, (0, 2, 1)))
        hidden = silu(gate_before) * hidden_before

        dhidden = jnp.einsum("bij,bjk->bik", jnp.transpose(w1, (0, 2, 1)), vi)
        dhidden_before = dhidden * silu(gate_before)
        dgate = dhidden * hidden_before
        dgate_before = silu_backprop(dgate, gate_before)

        dw1 = jnp.einsum("bij,bjk->bik", vi, jnp.transpose(hidden * lr1i, (0, 2, 1)))
        dw0 = jnp.einsum("bij,bjk->bik", dgate_before, ki * lr0i)
        dw2 = jnp.einsum("bij,bjk->bik", dhidden_before, ki * lr2i)

        if momentum is not None:
            m_i = jnp.mean(momentum[:, s:e, :], axis=1, keepdims=True)
            dw0 += dw0 * m_i
            dw1 += dw1 * m_i
            dw2 += dw2 * m_i

        if use_muon:
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw2 = zeropower_via_newtonschulz5(dw2)

        w0 = w0 + dw0
        w1 = w1 + dw1
        w2 = w2 + dw2

        w0 = w0 / (jnp.linalg.norm(w0, axis=2, keepdims=True) + 1e-5) * w0_norm
        w1 = w1 / (jnp.linalg.norm(w1, axis=2, keepdims=True) + 1e-5) * w1_norm
        w2 = w2 / (jnp.linalg.norm(w2, axis=2, keepdims=True) + 1e-5) * w2_norm

    s = l - chunk_size
    qi = q[:, :, s:]
    h = jnp.einsum("bij,bjk->bik", w2, qi)
    gate = silu(jnp.einsum("bij,bjk->bik", w0, qi))
    output = output.at[:, :, s:].set(jnp.einsum("bij,bjk->bik", w1, gate * h))
    return jnp.transpose(output, (0, 2, 1))


class CausalLaCTSwiGLUWithSlidingWindowAttn(nn.Module):
    dim: int
    head_dim: int
    attn_head_dim: int
    lact_chunk_size: int = 2048
    window_size: int = 2048
    inter_multi: float = 1.0
    use_o_norm: bool = True
    use_momentum: bool = True
    use_muon: bool = True
    base_lr: float = 1e-2
    ttt_scale_before_sum: bool = True

    def setup(self):
        d_h = int(self.head_dim * self.inter_multi)
        self.num_ttt_heads = self.dim // self.head_dim

        self.to_qkv = nn.Dense(3 * self.dim, use_bias=False)
        self.lr_proj = nn.Dense(3 * self.num_ttt_heads, use_bias=False)
        self.o_proj = nn.Dense(self.dim, use_bias=False)

        self.qk_scale = self.param("qk_scale", nn.initializers.ones, (self.dim, 2))
        self.qk_offset = self.param("qk_offset", nn.initializers.zeros, (self.dim, 2))

        self.w0 = self.param("w0", nn.initializers.normal(stddev=1.0 / math.sqrt(self.head_dim)), (self.num_ttt_heads, d_h, self.head_dim))
        self.w1 = self.param("w1", nn.initializers.normal(stddev=1.0 / math.sqrt(d_h)), (self.num_ttt_heads, self.head_dim, d_h))
        self.w2 = self.param("w2", nn.initializers.normal(stddev=1.0 / math.sqrt(self.head_dim)), (self.num_ttt_heads, d_h, self.head_dim))

        if self.use_momentum:
            self.momentum_proj = nn.Sequential([nn.Dense(self.num_ttt_heads), nn.sigmoid])

        if self.use_o_norm:
            self.o_norm = nn.LayerNorm()
        else:
            self.o_norm = lambda x: x

        if self.ttt_scale_before_sum:
            self.ttt_scale_proj = nn.Dense(self.num_ttt_heads)

        self.base_lr_inv = inv_softplus(self.base_lr)

    def __call__(self, x):
        qkv = self.to_qkv(x)
        qkv = silu(qkv)

        b, l, _ = x.shape
        h = self.num_ttt_heads
        d = self.head_dim

        qkv = qkv.reshape(b, l, h, 3 * d)
        ttt_q, ttt_k, ttt_v = jnp.split(qkv, 3, axis=-1)
        ttt_q = l2_norm(ttt_q, axis=-1).reshape(b * h, l, d)
        ttt_k = l2_norm(ttt_k, axis=-1).reshape(b * h, l, d)
        ttt_v = ttt_v.reshape(b * h, l, d)

        lr = jax.nn.softplus(self.lr_proj(x) + self.base_lr_inv)
        lr = lr.reshape(b, l, 3, h).transpose(2, 0, 3, 1).reshape(3, b * h, l, 1)
        lr0, lr1, lr2 = lr[0], lr[1], lr[2]

        w0 = jnp.tile(self.w0[None], (b, 1, 1))
        w1 = jnp.tile(self.w1[None], (b, 1, 1))
        w2 = jnp.tile(self.w2[None], (b, 1, 1))

        momentum = self.momentum_proj(x) if self.use_momentum else None
        if momentum is not None:
            momentum = momentum.reshape(b, l, h, 1).transpose(0, 2, 1, 3).reshape(b * h, l, 1)

        ttt_output = block_causal_lact_swiglu(w0, w1, w2, ttt_q, ttt_k, ttt_v, lr0, lr1, lr2, momentum, self.use_muon, self.lact_chunk_size)
        ttt_output = self.o_norm(ttt_output)

        if self.ttt_scale_before_sum:
            scale = silu(self.ttt_scale_proj(x)).reshape(b, l, h, 1).transpose(0, 2, 1, 3).reshape(b * h, l, 1)
            ttt_output *= scale

        ttt_output = ttt_output.reshape(b, h, l, d).transpose(0, 2, 1, 3).reshape(b, l, h * d)
        return self.o_proj(ttt_output)



def _test_jax_lact():
    key = jax.random.PRNGKey(0)
    B, L, D, HeadDim = 1, 2048*3, 2048, 512
    inter = 1
    d_h = int(HeadDim * inter)
    
    q = jax.random.normal(key, (B, L, HeadDim), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (B, L, HeadDim), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (B, L, HeadDim), dtype=jnp.bfloat16)
    lr0 = jnp.ones((B, L, 1), dtype=jnp.float32) * 1e-2
    lr1 = jnp.ones((B, L, 1), dtype=jnp.float32) * 1e-2
    lr2 = jnp.ones((B, L, 1), dtype=jnp.float32) * 1e-2

    w0 = jnp.ones((B, d_h, HeadDim), dtype=jnp.float32)
    w1 = jnp.ones((B, HeadDim, d_h), dtype=jnp.float32)
    w2 = jnp.ones((B, d_h, HeadDim), dtype=jnp.float32)

    output = block_causal_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, None)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    _test_jax_lact()
