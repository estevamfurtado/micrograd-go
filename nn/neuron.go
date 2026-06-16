package nn

import (
	"github.com/estevamfurtado/micrograd-go/engine"
)

type Neuron struct {
	weights    []*engine.Value
	bias       *engine.Value
	activation func(x *engine.Value) *engine.Value
}

func NewNeuron(in int, factory ParamsFactory, activation func(x *engine.Value) *engine.Value) *Neuron {
	bias := engine.Const(factory.BiasGenerator())
	weights := make([]*engine.Value, in)
	for i := 0; i < in; i++ {
		weights[i] = engine.Const(factory.WeightGenerator(in))
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
