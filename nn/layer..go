package nn

import "github.com/estevamfurtado/micrograd-go/engine"

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
