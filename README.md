# micrograd-go

A Go reimplementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a learning project to build autograd, backpropagation, and neural networks from scratch.

## References

- Original repo: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- Lecture: [micrograd explained - backpropagation and training neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)

## What is micrograd?

A scalar autograd engine (~100 lines) plus a minimal neural network library (~50 lines) with a PyTorch-like API. Every operation is a scalar; the computation graph is built dynamically and gradients flow via reverse-mode backpropagation.

## Project structure

```
micrograd-go/
├── engine/          # autograd — port of engine.py (Value, ops, backward)
├── nn/              # neural nets — port of nn.py (Neuron, Layer, MLP)
├── datasets/        # toy datasets (make_moons, JSONL I/O)
└── examples/moons/  # train an MLP on the moons dataset (demo.ipynb equivalent)
```

## Quick start

```bash
go build ./...
go test ./...

# generate dataset, export JSONL, plot points, and train the MLP
go run ./examples/moons/
```

Run from the repo root so paths like `examples/moons/moons.jsonl` resolve correctly.

## What's implemented

- **engine** — `Value`, `Add`, `Mul`, `Pow`, `Div`, `Neg`, `ReLU`, `Backward()`
- **nn** — `Neuron`, `Layer`, `MLP` with ReLU hidden layers and a linear output layer
- **datasets** — `make_moons`, JSONL read/write
- **examples/moons** — hinge loss, L2 regularization, SGD with learning-rate decay

## Roadmap

- [ ] decision boundary plot
- [ ] gradient checking with finite differences
- [ ] `Tanh` activation (optional)

## License

MIT — same educational spirit as the original project.
