package loss

import (
	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn/train"
)

// BinaryMarginCalculator implements max(0, 1 - y * label) for single-logit binary classification.
type BinaryMarginCalculator struct{}

func (l *BinaryMarginCalculator) Calculate(logits []*engine.Value, sample train.Sample) *engine.Value {
	y := logits[0]
	return engine.ReLU(engine.Add(engine.Const(1), engine.Neg(engine.Mul(y, engine.Const(sample.Y)))))
}

func (l *BinaryMarginCalculator) IsAccurate(logits []*engine.Value, sample train.Sample) bool {
	y := logits[0]
	return (y.Data > 0) == (sample.Y > 0)
}
