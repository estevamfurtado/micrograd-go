package nn

import (
	"math/rand"

	"github.com/estevamfurtado/micrograd-go/engine"
)

func randomWeight() float64 {
	return rand.Float64()*2 - 1
}

func linear(x *engine.Value) *engine.Value {
	return x
}

// Neuron

type Neuron struct {
	weights    []*engine.Value
	bias       *engine.Value
	activation func(x *engine.Value) *engine.Value
}

func NewNeuron(in int, nonlin bool) *Neuron {
	activation := engine.ReLU
	if !nonlin {
		activation = linear
	}

	bias := engine.Const(randomWeight())
	weights := make([]*engine.Value, in)
	for i := 0; i < in; i++ {
		weights[i] = engine.Const(randomWeight())
	}
	return &Neuron{
		weights:    weights,
		bias:       bias,
		activation: activation,
	}
}

func (n *Neuron) Calculate(inputs []*engine.Value) *engine.Value {
	if len(inputs) != len(n.weights) {
		panic("number of inputs does not match number of weights")
	}

	values := make([]*engine.Value, len(inputs)+1)
	values[0] = n.bias
	for i := range inputs {
		values[i+1] = engine.Mul(inputs[i], n.weights[i])
	}

	return n.activation(engine.Add(values...))
}

func (n *Neuron) Parameters() []*engine.Value {
	return append(n.weights, n.bias)
}

// Layer

type Layer struct {
	neurons []*Neuron
}

func NewLayer(in, out int, nonlin bool) *Layer {
	neurons := make([]*Neuron, out)
	for i := range neurons {
		neurons[i] = NewNeuron(in, nonlin)
	}
	return &Layer{neurons: neurons}
}

func (l *Layer) Calculate(inputs []*engine.Value) []*engine.Value {
	outputs := make([]*engine.Value, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Calculate(inputs)
	}
	return outputs
}

func (l *Layer) Parameters() []*engine.Value {
	params := []*engine.Value{}
	for _, neuron := range l.neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

//  Multi-Layer Perceptron

type MLP struct {
	layers []*Layer
}

func NewMLP(in int, sizes []int) *MLP {
	layers := make([]*Layer, len(sizes))
	prev := in
	for i, out := range sizes {
		nonlin := i != len(sizes)-1
		layers[i] = NewLayer(prev, out, nonlin)
		prev = out
	}

	return &MLP{layers: layers}
}

func (m *MLP) Calculate(inputs []*engine.Value) []*engine.Value {
	outputs := inputs
	for _, layer := range m.layers {
		outputs = layer.Calculate(outputs)
	}
	return outputs
}

func (m *MLP) Parameters() []*engine.Value {
	params := []*engine.Value{}
	for _, layer := range m.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
