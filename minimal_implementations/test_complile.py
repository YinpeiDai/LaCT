import time
import torch


# @torch.compile()
# def my_func(x):
#     # print("x:", x)
#     return x @ x.T

@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    print(dx[0, 0])
    return dx


# warmup
for i in range(10):
    dy = torch.randn(10000, 10000, device="cuda")
    x = torch.randn(10000, 10000, device="cuda")
    silu_backprop(dy, x)
    end = time.time()


start = time.time()
for i in range(10000):
    dy = torch.randn(10000, 10000, device="cuda")
    x = torch.randn(10000, 10000, device="cuda")
    silu_backprop(dy, x)
end = time.time()
print(f"Time taken: {end - start} seconds")