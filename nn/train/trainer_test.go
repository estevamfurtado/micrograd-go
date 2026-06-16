package train

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
)

func TestTrainer_writesRunLogs(t *testing.T) {
	dir := t.TempDir()

	data := Samples{
		{X: []float64{0, 0}, Y: -1},
		{X: []float64{1, 1}, Y: 1},
		{X: []float64{0, 1}, Y: -1},
		{X: []float64{1, 0}, Y: 1},
	}

	in := nn.NewLayer(2, 2, RandomInit, ReLUActivation)
	out := nn.NewLayer(2, 1, RandomInit, LinearActivation)
	model := nn.NewMLP().AddLayer(in).AddLayer(out)

	cfg := Config{
		HiddenSize: 2,
		Epochs:     1,
		BatchSize:  2,
		LR:         0.1,
		RunDir:     dir,
	}

	trainer := NewTrainer(model, cfg, &fixedLoss{}, data)
	trainer.Train(data)

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	require.Len(t, entries, 1)

	runDir := filepath.Join(dir, entries[0].Name())
	_, err = os.Stat(filepath.Join(runDir, "summary.txt"))
	require.NoError(t, err)
	_, err = os.Stat(filepath.Join(runDir, "epoch_0.txt"))
	require.NoError(t, err)

	logs, err := os.ReadFile(filepath.Join(runDir, "logs.txt"))
	require.NoError(t, err)
	assert.Contains(t, string(logs), "Epoch 0")
}

type fixedLoss struct{}

func (fixedLoss) Calculate(logits []*engine.Value, sample Sample) *engine.Value {
	y := logits[0]
	return engine.ReLU(engine.Add(engine.Const(1), engine.Neg(engine.Mul(y, engine.Const(sample.Y)))))
}

func (fixedLoss) IsAccurate(logits []*engine.Value, sample Sample) bool {
	y := logits[0]
	return (y.Data > 0) == (sample.Y > 0)
}
