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
└── examples/moons/      # moons demo (MakeMoons, JSONL, train, plot)
    └── moons.jsonl      # committed dataset (100 samples, seed 1337)
```

## Quick start

```bash
go build ./...
go test ./...

# load moons.jsonl and train the MLP (100 epochs, full batch)
go run ./examples/moons/
```

Run from the **repo root** — paths like `examples/moons/moons.jsonl` are relative to the working directory.

The moons example loads the committed JSONL file, trains a `2 → 16 → 16 → 1` MLP, and saves a decision boundary plot to `examples/moons/decision_boundary.png`.

To regenerate the dataset and scatter plot (`moons.png`), call `GenerateDataset(100, 0.1)` from `examples/moons/dataset.go` (not wired into `main` by default).

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
