// Package nn provides a minimal neural-network library on top of engine —
// the Go port of micrograd's nn.py.
//
// Planned contents:
//   - Module: base type with Parameters() for SGD updates
//   - Neuron, Layer, MLP: forward pass built from engine.Value
package nn
