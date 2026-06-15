package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
)

// Config holds all training settings. Edit these values directly.
type Config struct {
	HiddenSize int     // hidden layer size
	Limit      int     // max training rows (0 = full 60k)
	TestLimit  int     // max test rows for eval (0 = full 10k)
	Epochs     int     // training epochs
	BatchSize  int     // batch size
	LR         float64 // initial learning rate (decays per epoch)
}

var config = Config{
	HiddenSize: 16,
	Limit:      60_000,
	TestLimit:  10_000,
	Epochs:     10,
	BatchSize:  32,
	LR:         0.02,
}

func linear(x *engine.Value) *engine.Value {
	return x
}

func main() {
	if err := ensureData(); err != nil {
		fmt.Fprintf(os.Stderr, "download: %v\n", err)
		os.Exit(1)
	}

	trainPath := filepath.Join(dataDir(), "mnist_train.csv")
	testPath := filepath.Join(dataDir(), "mnist_test.csv")

	train, err := LoadCSV(trainPath, config.Limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load train: %v\n", err)
		os.Exit(1)
	}
	test, err := LoadCSV(testPath, config.TestLimit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load test: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("train: %d samples, test: %d samples\n", len(train), len(test))

	hidden := nn.NewLayer(numPixels, config.HiddenSize, nn.HeInit, engine.ReLU)
	out := nn.NewLayer(config.HiddenSize, numClasses, nn.XavierInit, linear)

	model := nn.NewMLP(hidden, out)
	fmt.Printf("model: %d parameters\n", len(model.Parameters()))

	loss := &CrossEntropyCalculator{}
	trainer := NewTrainer(model, config.LR, config.Epochs, config.BatchSize, loss, test)
	trainer.Train(train)
}
