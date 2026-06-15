package main

import (
	"fmt"
	"math/rand"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
)

type LossCalculator interface {
	Calculate(logits []*engine.Value, sample Sample) *engine.Value
	IsAccurate(logits []*engine.Value, sample Sample) bool
}

type Trainer struct {
	model      *nn.MLP
	lr         float64
	epochs     int
	batch_size int
	lossCalc   LossCalculator
	testData   Samples
}

func NewTrainer(model *nn.MLP, lr float64, epochs, batchSize int, lossCalc LossCalculator, testData Samples) *Trainer {
	return &Trainer{model: model, lr: lr, epochs: epochs, batch_size: batchSize, lossCalc: lossCalc, testData: testData}
}

func (t *Trainer) SampleInputs(sample Sample) []*engine.Value {
	inputs := make([]*engine.Value, numPixels)
	for i, pix := range sample.X {
		inputs[i] = engine.Const(pix)
	}
	return inputs
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

func (t *Trainer) Train(data Samples) {
	testAccuracy := t.Accuracy(t.testData)
	fmt.Printf("initial test accuracy: %.1f%%\n", testAccuracy*100)

	rng := rand.New(rand.NewSource(1337))
	for epoch := 0; epoch < t.epochs; epoch++ {
		data.Shuffle(rng)
		batches := len(data) / t.batch_size
		lr := t.learningRate(epoch)

		fmt.Printf("epoch %d/%d: learning rate %.3f\n", epoch, t.epochs, lr)

		for batch := 0; batch < batches; batch++ {
			batchData := data[batch*t.batch_size : (batch+1)*t.batch_size]

			t.zeroGrad()
			loss, accuracy := t.loss(batchData)

			if batch%50 == 0 {
				fmt.Printf("\tbatch %d/%d: loss %.2f, accuracy %.1f%%\n", batch, batches, loss.Data, accuracy*100)
			}

			loss.Backward()

			for _, p := range t.model.Parameters() {
				p.Data -= lr * p.Grad
			}
		}

		testAccuracy = t.Accuracy(t.testData)
		trainAccuracy := t.Accuracy(data)
		fmt.Printf("test: %.1f%%, train: %.1f%%\n", testAccuracy*100, trainAccuracy*100)
	}

}

func (t *Trainer) loss(batchData []Sample) (*engine.Value, float64) {
	losses := make([]*engine.Value, 0, len(batchData))
	score := 0

	for _, sample := range batchData {
		logits := t.model.Calculate(t.SampleInputs(sample))
		loss := t.lossCalc.Calculate(logits, sample)
		losses = append(losses, loss)
		if t.lossCalc.IsAccurate(logits, sample) {
			score++
		}
	}

	dataLoss := engine.Mul(engine.Add(losses...), engine.Const(1.0/float64(len(losses))))
	regularizationLoss := t.computeRegularizationLoss(t.model)
	totalLoss := engine.Add(dataLoss, regularizationLoss)

	return totalLoss, float64(score) / float64(len(batchData))
}

func (t *Trainer) computeRegularizationLoss(model *nn.MLP) *engine.Value {
	alpha := 1e-4
	parameters := model.Parameters()
	regLosses := make([]*engine.Value, 0, len(parameters))
	for _, parameter := range parameters {
		regLosses = append(regLosses, engine.Mul(parameter, parameter))
	}
	return engine.Mul(engine.Add(regLosses...), engine.Const(alpha))
}

// Accuracy runs forward-only classification accuracy on data.
func (t *Trainer) Accuracy(data Samples) float64 {
	if len(data) == 0 {
		return 0
	}
	score := 0
	for _, sample := range data {
		logits := t.model.Calculate(t.SampleInputs(sample))
		if t.lossCalc.IsAccurate(logits, sample) {
			score++
		}
	}
	return float64(score) / float64(len(data))
}
