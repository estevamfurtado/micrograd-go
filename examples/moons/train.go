package main

import (
	"fmt"
	"math/rand"

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
	if t.epochs <= 1 {
		return t.lr
	}
	return t.lr * (1.0 - 0.9*float64(epoch)/float64(t.epochs))
}

func (t *Trainer) zeroGrad() {
	for _, p := range t.model.Parameters() {
		p.Grad = 0
	}
}

func (t *Trainer) Train(data datasets.Samples) {
	rng := rand.New(rand.NewSource(1337))

	step := 1

	for epoch := 0; epoch < t.epochs; epoch++ {
		data.Shuffle(rng)

		batches := len(data) / t.batch_size
		lr := t.learningRate(epoch)
		for batch := 0; batch < batches; batch++ {
			batch_data := data[batch*t.batch_size : (batch+1)*t.batch_size]

			t.zeroGrad()
			loss, accuracy := t.loss(batch_data)
			fmt.Printf("step %d loss %f, accuracy %f%%\n", step, loss.Data, accuracy*100)

			loss.Backward()

			for _, p := range t.model.Parameters() {
				p.Data -= lr * p.Grad
			}

			step++
		}
	}
}

func (t *Trainer) loss(batch_data []datasets.Sample) (*engine.Value, float64) {
	score := 0

	// Data Loss
	losses := make([]*engine.Value, 0, len(batch_data))
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
	regLosses := make([]*engine.Value, 0, len(parameters))
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
