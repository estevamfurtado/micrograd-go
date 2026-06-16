package train

import "github.com/estevamfurtado/micrograd-go/engine"

func LinearActivation(x *engine.Value) *engine.Value {
	return x
}

func ReLUActivation(x *engine.Value) *engine.Value {
	return engine.ReLU(x)
}
