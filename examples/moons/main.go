package main

import (
	"fmt"

	"github.com/estevamfurtado/micrograd-go/nn"
)

func main() {
	data := loadMoons()
	fmt.Printf("loaded %d samples\n", len(data))

	model := nn.NewMLP(2, []int{16, 16, 1}) // 2-layer neural network
	fmt.Printf("model has %d parameters\n", len(model.Parameters()))

	// i only have 100 samples
	lr := 1.0 // matches Karpathy's demo schedule (decays to ~0.1 over 100 epochs)
	epochs := 100
	batch_size := len(data)
	trainer := NewTrainer(model, lr, epochs, batch_size)
	trainer.Train(data)

	plotDecisionBoundaryOrExit(model, data)
}
