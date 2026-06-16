package train

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
)

type Config struct {
	HiddenSize int     // hidden layer size
	Limit      int     // max training rows (0 = full 60k)
	TestLimit  int     // max test rows for eval (0 = full 10k)
	Epochs     int     // training epochs
	BatchSize  int     // batch size
	LR         float64 // initial learning rate (decays per epoch)
	RunDir     string  // base directory for structured run logs (empty = disabled)
}

type LossCalculator interface {
	Calculate(logits []*engine.Value, sample Sample) *engine.Value
	IsAccurate(logits []*engine.Value, sample Sample) bool
}

type Trainer struct {
	model    *nn.MLP
	config   Config
	lossCalc LossCalculator
	testData []Sample
	runLog   *RunLogger
}

func NewTrainer(model *nn.MLP, config Config, lossCalc LossCalculator, testData []Sample) *Trainer {
	return &Trainer{model: model, config: config, lossCalc: lossCalc, testData: testData}
}

func (t *Trainer) SampleInputs(sample Sample) []*engine.Value {
	inputs := make([]*engine.Value, len(sample.X))
	for i, pix := range sample.X {
		inputs[i] = engine.Const(pix)
	}
	return inputs
}

func (t *Trainer) learningRate(epoch int) float64 {
	if t.config.Epochs <= 1 {
		return t.config.LR
	}
	return t.config.LR * (1.0 - 0.9*float64(epoch)/float64(t.config.Epochs))
}

func (t *Trainer) logHyperparams() {
	c := t.config
	fmt.Printf("hidden_size=%d limit=%d test_limit=%d epochs=%d batch_size=%d lr=%g\n",
		c.HiddenSize, c.Limit, c.TestLimit, c.Epochs, c.BatchSize, c.LR)
}

func minutesSince(start time.Time) float64 {
	return time.Since(start).Minutes()
}

func (t *Trainer) zeroGrad() {
	for _, p := range t.model.Parameters() {
		p.Grad = 0
	}
}

const logEveryNBatches = 50

func (t *Trainer) Train(data Samples) {
	if t.config.RunDir != "" && t.runLog == nil {
		runLog, err := NewRunLogger(t.config.RunDir, t.config)
		if err != nil {
			fmt.Fprintf(os.Stderr, "run log: %v\n", err)
		} else {
			t.runLog = runLog
			fmt.Printf("run logs: %s\n", runLog.Dir())
		}
	}

	t.logHyperparams()
	trainingStart := time.Now()
	fmt.Printf("training started (%.2f min)\n", minutesSince(trainingStart))

	testAccuracy := t.Accuracy(t.testData)
	fmt.Printf("initial test accuracy: %.1f%%\n", testAccuracy*100)
	if t.runLog != nil {
		if err := t.runLog.RecordInitial(testAccuracy); err != nil {
			fmt.Fprintf(os.Stderr, "run log: %v\n", err)
		}
	}

	fmt.Printf("%5s | %5s | %6s | %6s | %8s\n", "epoch", "batch", "min", "loss", "accuracy")

	rng := rand.New(rand.NewSource(1337))
	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		fmt.Printf("epoch %d started (%.2f min, lr %.3f)\n", epoch, minutesSince(trainingStart), t.learningRate(epoch))

		data.Shuffle(rng)
		batches := len(data) / t.config.BatchSize
		lr := t.learningRate(epoch)
		var epochBatches []runBatchRow
		var lastLoss float64
		var lastAccuracy float64

		for batch := 0; batch < batches; batch++ {
			batchData := data[batch*t.config.BatchSize : (batch+1)*t.config.BatchSize]

			t.zeroGrad()
			loss, accuracy := t.loss(batchData)
			lastLoss = loss.Data
			lastAccuracy = accuracy

			if batch%logEveryNBatches == 0 {
				minutes := minutesSince(trainingStart)
				fmt.Printf("%5d | %5d | %6.2f | %6.2f | %7.1f%%\n",
					epoch, batch, minutes, loss.Data, accuracy*100)
				epochBatches = append(epochBatches, runBatchRow{
					batch:    batch,
					minutes:  minutes,
					loss:     loss.Data,
					accuracy: accuracy,
				})
			}

			loss.Backward()

			for _, p := range t.model.Parameters() {
				p.Data -= lr * p.Grad
			}
		}

		if len(epochBatches) == 0 && batches > 0 {
			epochBatches = append(epochBatches, runBatchRow{
				batch:    batches - 1,
				minutes:  minutesSince(trainingStart),
				loss:     lastLoss,
				accuracy: lastAccuracy,
			})
		}

		testAccuracy = t.Accuracy(t.testData)
		trainAccuracy := t.Accuracy(data)
		fmt.Printf("epoch %d done (%.2f min): test %.1f%%, train %.1f%%\n",
			epoch, minutesSince(trainingStart), testAccuracy*100, trainAccuracy*100)

		if t.runLog != nil {
			if err := t.runLog.RecordEpoch(epoch, lr, testAccuracy, trainAccuracy, epochBatches); err != nil {
				fmt.Fprintf(os.Stderr, "run log: %v\n", err)
			}
		}
	}

	fmt.Printf("training finished (%.2f min)\n", minutesSince(trainingStart))
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
