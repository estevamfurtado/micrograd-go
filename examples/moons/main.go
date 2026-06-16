package main

import (
	"fmt"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
)

func linear(x *engine.Value) *engine.Value {
	return x
}

func main() {
	data := loadMoons()
	fmt.Printf("loaded %d samples\n", len(data))

	in := nn.NewLayer(2, 16, nn.RandomInit, engine.ReLU)
	h1 := nn.NewLayer(16, 16, nn.RandomInit, engine.ReLU)
	out := nn.NewLayer(16, 1, nn.RandomInit, linear)

	model := nn.NewMLP().AddLayer(in).AddLayer(h1).AddLayer(out)
	fmt.Printf("model has %d parameters\n", len(model.Parameters()))

	// i only have 100 samples
	lr := 1.0 // matches Karpathy's demo schedule (decays to ~0.1 over 100 epochs)
	epochs := 100
	batch_size := len(data)
	trainer := NewTrainer(model, lr, epochs, batch_size)
	trainer.Train(data)

	plotDecisionBoundaryOrExit(model, data)
}
