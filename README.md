# 🧠 Micrograd From Scratch

A scalar-valued autograd engine and neural network library built from scratch in pure Python — inspired by [Andrej Karpathy's micrograd lecture](https://youtu.be/VMj-3S1tku0).

## What's inside

- **Autograd engine** — `Value` class with full backward pass via topological sort
- **Operations** — `+`, `*`, `**`, `exp`, `log`, `tanh`, `sigmoid`, `relu`, `leaky_relu`, `swish`
- **Neural network** — `Neuron`, `Layer`, `MLP` built on top of the engine
- **Loss functions** — MSE, MAE, Hinge Loss, Binary Cross Entropy
- **Batched training** — `DataLoader` with shuffle support
- **Graph visualization** — renders computation graphs inline in Jupyter
- **Interactive UI** — train and predict live using `ipywidgets`

## Example

```python
from micrograd import Value, MLP, Loss

model = MLP(3, [4, 4, 1])

xs = [[Value(2.0), Value(3.0), Value(-1.0)], ...]
ys = [Value(1.0), Value(-1.0), ...]

for epoch in range(20):
    ypred = [model(x) for x in xs]
    loss = Loss.mse(ypred, ys)
    for p in model.parameters(): p.grad = 0.0
    loss.backward()
    for p in model.parameters(): p.data -= 0.05 * p.grad
```

## Concepts covered

- Forward pass & computation graphs
- Chain rule & backpropagation
- Topological sort for gradient flow
- Gradient descent optimization
- Neural network architecture

## Credits

Inspired by [Andrej Karpathy](https://github.com/karpathy/micrograd)'s legendary lecture on backpropagation.
