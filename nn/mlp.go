package nn

import "github.com/estevamfurtado/micrograd-go/engine"

type MLP struct {
	layers []*Layer
}

func NewMLP(layers ...*Layer) *MLP {
	for i := 0; i < len(layers)-1; i++ {
		if layers[i].out != layers[i+1].in {
			panic("number of inputs does not match number of outputs")
		}
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
