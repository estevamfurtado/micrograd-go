# micrograd-go

A Go reimplementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## References

- Original repo: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- Lecture: [micrograd explained - backpropagation and training neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)

## What is micrograd?

A scalar autograd engine plus a minimal neural network library with a PyTorch-like API. Every operation is a scalar; the computation graph is built dynamically and gradients flow via reverse-mode backpropagation.

## Project structure

```
micrograd-go/
├── engine/              # autograd — port of engine.py (Value, ops, backward)
├── nn/                  # neural nets — port of nn.py (Neuron, Layer, MLP)
└── examples/
    ├── mnist/sample/data/   # MNIST CSVs (downloaded on first run, gitignored)
    └── moons/sample/data/   # moons.jsonl (generated on first run, gitignored)
```

## Quick start

```bash
go build ./...
go test ./...

# load moons data and train the MLP (100 epochs, full batch)
go run ./examples/moons/
```

Data files live under `examples/*/sample/data/` and are created automatically on first run (`EnsureData`).

The moons example trains a `2 → 16 → 16 → 1` MLP and saves a decision boundary plot to `decision_boundary.png` in the moons example directory.

To regenerate the dataset and scatter plot (`moons.png`), call `sample.GenerateDataset(100, 0.1)` (not wired into `main` by default).

## What's implemented

- **engine** — `Value`, `Add`, `Mul`, `Pow`, `Div`, `Neg`, `ReLU`, `Backward()`
- **nn** — `Neuron`, `Layer`, `MLP` with ReLU hidden layers and a linear output layer
- **examples/moons** — `MakeMoons`, JSONL I/O, training loop (hinge loss, L2 reg, decaying LR)
- **tests** — engine ops/backward, moons JSONL round-trip

## From the author

Why not Python? 
1. I started the learning over Python, but honestly, its hard not to copypaste when you are writing in the original lang.
2. I like Go. Typechecks, ergonomics, etc.

Along the way, [Cursor](https://cursor.com) agents helped with the boring stuff — debugging `make`+`append` slice crimes, translating `plt.contourf` into gonum heatmaps. The autograd, the training loop, and the questionable choices were mine.

## License

MIT — same educational spirit as the original project.
