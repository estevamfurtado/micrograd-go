package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/estevamfurtado/micrograd-go/examples/mnist/sample"
	"github.com/estevamfurtado/micrograd-go/nn"
	"github.com/estevamfurtado/micrograd-go/nn/train"
	"github.com/estevamfurtado/micrograd-go/nn/train/loss"
)

func main() {
	if err := sample.EnsureData(); err != nil {
		fmt.Fprintf(os.Stderr, "download: %v\n", err)
		os.Exit(1)
	}

	trainPath := filepath.Join(sample.DataDir(), "mnist_train.csv")
	testPath := filepath.Join(sample.DataDir(), "mnist_test.csv")

	cfg := train.Config{
		HiddenSize: 16,
		Limit:      60_000,
		TestLimit:  10_000,
		Epochs:     10,
		BatchSize:  32,
		LR:         0.02,
		RunDir:     "runs",
	}

	trd, err := sample.LoadCSV(trainPath, cfg.Limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load train: %v\n", err)
		os.Exit(1)
	}
	ttd, err := sample.LoadCSV(testPath, cfg.TestLimit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load test: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("train: %d samples, test: %d samples\n", len(trd), len(ttd))

	input := nn.NewLayer(sample.NumPixels, cfg.HiddenSize, train.HeInit, train.ReLUActivation)
	out := nn.NewLayer(cfg.HiddenSize, sample.NumClasses, train.XavierInit, train.LinearActivation)

	model := nn.NewMLP().AddLayer(input).AddLayer(out)

	fmt.Printf("model: %d parameters\n", len(model.Parameters()))

	loss := &loss.CrossEntropyCalculator{}
	trainer := train.NewTrainer(model, cfg, loss, ttd)
	trainer.Train(trd)
}
