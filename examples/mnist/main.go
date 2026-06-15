package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/estevamfurtado/micrograd-go/nn"
)

// Config holds all training settings. Edit these values directly.
type Config struct {
	Limit     int     // max training rows (0 = full 60k)
	TestLimit int     // max test rows for eval (0 = full 10k)
	Epochs    int     // training epochs
	BatchSize int     // batch size
	LR        float64 // initial learning rate (decays per epoch)
}

var config = Config{
	Limit:     1000,
	TestLimit: 500,
	Epochs:    5,
	BatchSize: 32,
	LR:        0.05,
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

	model := nn.NewMLP(numPixels, []int{16, numClasses})
	fmt.Printf("model: %d parameters\n", len(model.Parameters()))

	loss := &CrossEntropyCalculator{}
	trainer := NewTrainer(model, config.LR, config.Epochs, config.BatchSize, loss)

	// initial accuracy
	initAcc := trainer.Accuracy(train)
	fmt.Printf("initial train accuracy: %.1f%% (expect ~10%%)\n", initAcc*100)

	// train
	trainer.Train(train)

	// final accuracy
	trainAcc := trainer.Accuracy(train)
	testAcc := trainer.Accuracy(test)
	fmt.Printf("final train accuracy: %.1f%%\n", trainAcc*100)
	fmt.Printf("final test accuracy:  %.1f%%\n", testAcc*100)
}
