package main

import (
	"fmt"

	"github.com/estevamfurtado/micrograd-go/examples/moons/sample"
	"github.com/estevamfurtado/micrograd-go/nn"
	"github.com/estevamfurtado/micrograd-go/nn/train"
	"github.com/estevamfurtado/micrograd-go/nn/train/loss"
)

func main() {
	data := sample.LoadMoons()
	fmt.Printf("loaded %d samples\n", len(data))

	in := nn.NewLayer(2, 16, train.RandomInit, train.ReLUActivation)
	h1 := nn.NewLayer(16, 16, train.RandomInit, train.ReLUActivation)
	out := nn.NewLayer(16, 1, train.RandomInit, train.LinearActivation)

	model := nn.NewMLP().AddLayer(in).AddLayer(h1).AddLayer(out)
	fmt.Printf("model has %d parameters\n", len(model.Parameters()))

	cfg := train.Config{
		HiddenSize: 16,
		Epochs:     100,
		BatchSize:  len(data),
		LR:         1.0,
	}

	trainer := train.NewTrainer(model, cfg, &loss.BinaryMarginCalculator{}, nil)
	trainer.Train(data)

	sample.PlotDecisionBoundaryOrExit(model, data)
}
