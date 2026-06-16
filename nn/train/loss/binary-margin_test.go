package loss

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn/train"
)

func TestBinaryMarginCalculator(t *testing.T) {
	calc := &BinaryMarginCalculator{}

	t.Run("correct prediction has zero loss", func(t *testing.T) {
		logits := []*engine.Value{engine.Const(2.0)}
		sample := train.Sample{Y: 1}
		loss := calc.Calculate(logits, sample)
		assert.InDelta(t, 0, loss.Data, 1e-9)
		assert.True(t, calc.IsAccurate(logits, sample))
	})

	t.Run("wrong sign has positive loss", func(t *testing.T) {
		logits := []*engine.Value{engine.Const(-0.5)}
		sample := train.Sample{Y: 1}
		loss := calc.Calculate(logits, sample)
		assert.InDelta(t, 1.5, loss.Data, 1e-9)
		assert.False(t, calc.IsAccurate(logits, sample))
	})
}
