package main

import (
	"fmt"

	"github.com/estevamfurtado/micrograd-go/datasets"
	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
)

type Trainer struct {
	model *nn.MLP
	// hyperparameters
	lr         float64 // learning rate: how much we'll update the model's parameters
	epochs     int     // number of epochs: how many times we'll iterate over the entire dataset
	batch_size int     // batch size: how many samples we'll use to compute the loss and accuracy
}

func NewTrainer(model *nn.MLP, lr float64, epochs int, batch_size int) *Trainer {
	return &Trainer{model: model, lr: lr, epochs: epochs, batch_size: batch_size}
}

func (t *Trainer) learningRate(epoch int) float64 {
	return 1.0 - 0.9*float64(epoch)/100
}

func (t *Trainer) Train(data datasets.Samples) {
	data.Shuffle()

	for epoch := 0; epoch < t.epochs; epoch++ {
		batches := len(data) / t.batch_size
		lr := t.learningRate(epoch)
		fmt.Printf("epoch %d of %d (lr %f) batches %d", epoch, t.epochs, lr, batches)
		for batch := 0; batch < batches; batch++ {
			fmt.Printf("batch %d: ", batch)

			batch_data := data[batch*t.batch_size : (batch+1)*t.batch_size]

			// forward pass
			loss, accuracy := t.loss(batch_data)
			fmt.Printf("loss %f, accuracy %f%%\n", loss.Data, accuracy*100)

			loss.Backward()

			for _, p := range t.model.Parameters() {
				p.Data -= lr * p.Grad
			}
		}
	}
}

func (t *Trainer) loss(batch_data []datasets.Sample) (*engine.Value, float64) {
	score := 0

	// Data Loss
	losses := make([]*engine.Value, len(batch_data))
	for _, sample := range batch_data {
		// forward pass
		inputs := []*engine.Value{
			engine.Const(sample.X[0]),
			engine.Const(sample.X[1]),
		}
		y := t.model.Calculate(inputs)[0]

		loss := singleLoss(y, sample.Y)
		losses = append(losses, loss)

		if (y.Data > 0) == (sample.Y > 0) {
			score++
		}
	}
	dataLoss := engine.Mul(engine.Add(losses...), engine.Const(1.0/float64(len(losses))))

	// Regularization Loss
	alpha := 1e-4
	parameters := t.model.Parameters()
	regLosses := make([]*engine.Value, len(parameters))
	for _, parameter := range parameters {
		regLosses = append(regLosses, engine.Mul(parameter, parameter))
	}
	regLoss := engine.Mul(engine.Add(regLosses...), engine.Const(alpha))

	// Total Loss
	totalLoss := engine.Add(dataLoss, regLoss)

	return totalLoss, float64(score) / float64(len(batch_data))
}

func singleLoss(y *engine.Value, sample_y float64) *engine.Value {
	return engine.ReLU(engine.Add(engine.Const(1), engine.Neg(engine.Mul(y, engine.Const(sample_y)))))
}
