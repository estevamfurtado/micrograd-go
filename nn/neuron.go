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
