package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/estevamfurtado/micrograd-go/nn"
)

func main() {
	limit := flag.Int("limit", 1000, "max training rows (0 = full 60k)")
	testLimit := flag.Int("test-limit", 500, "max test rows for eval (0 = full 10k)")
	epochs := flag.Int("epochs", 5, "training epochs")
	batchSize := flag.Int("batch", 32, "batch size")
	lr := flag.Float64("lr", 0.05, "initial learning rate (decays per epoch)")
	flag.Parse()

	if err := ensureData(); err != nil {
		fmt.Fprintf(os.Stderr, "download: %v\n", err)
		os.Exit(1)
	}

	trainPath := filepath.Join(dataDir(), "mnist_train.csv")
	testPath := filepath.Join(dataDir(), "mnist_test.csv")

	train, err := LoadCSV(trainPath, *limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load train: %v\n", err)
		os.Exit(1)
	}
	test, err := LoadCSV(testPath, *testLimit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load test: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("train: %d samples, test: %d samples\n", len(train), len(test))

	model := nn.NewMLP(numPixels, []int{16, numClasses})
	fmt.Printf("model: %d parameters\n", len(model.Parameters()))

	loss := &CrossEntropyCalculator{}
	trainer := NewTrainer(model, *lr, *epochs, *batchSize, loss)

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
