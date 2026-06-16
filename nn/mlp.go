package nn

import "github.com/estevamfurtado/micrograd-go/engine"

type MLP struct {
	layers []*Layer
}

func NewMLP() *MLP {
	return &MLP{layers: []*Layer{}}
}

func (m *MLP) AddLayer(layer *Layer) *MLP {
	lastLayer := m.layers[len(m.layers)-1]
	if lastLayer.out != layer.in {
		panic("number of inputs does not match number of outputs")
	}

	m.layers = append(m.layers, layer)
	return m
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
