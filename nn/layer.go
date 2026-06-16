package nn

import "github.com/estevamfurtado/micrograd-go/engine"

type Layer struct {
	in      int
	out     int
	neurons []*Neuron
}

type ParamsFactory struct {
	BiasGenerator   func() float64
	WeightGenerator func(fanIn int) float64
}

func NewLayer(in, out int, factory ParamsFactory, activation func(x *engine.Value) *engine.Value) *Layer {
	neurons := make([]*Neuron, out)
	for i := range neurons {
		neurons[i] = NewNeuron(in, factory, activation)
	}
	return &Layer{in: in, out: out, neurons: neurons}
}

func (l *Layer) Calculate(inputs []*engine.Value) []*engine.Value {
	if len(inputs) != len(l.neurons[0].weights) {
		panic("number of inputs does not match number of weights")
	}

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
