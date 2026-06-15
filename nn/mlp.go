package nn

import "github.com/estevamfurtado/micrograd-go/engine"

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
