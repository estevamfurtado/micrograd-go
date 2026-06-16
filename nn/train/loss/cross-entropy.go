package loss

import (
	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn/train"
)

type CrossEntropyCalculator struct{}

func OneHot(label int, numClasses int) []float64 {
	v := make([]float64, numClasses)
	v[label] = 1
	return v
}

func (l *CrossEntropyCalculator) Calculate(logits []*engine.Value, sample train.Sample) *engine.Value {
	probs := Softmax(logits)
	return crossEntropyOneHot(probs, oneHot(int(sample.Y), len(logits)))
}

func (l *CrossEntropyCalculator) IsAccurate(logits []*engine.Value, sample train.Sample) bool {
	answer := 0
	for i, l := range logits {
		if l.Data > logits[answer].Data {
			answer = i
		}
	}

	return answer == int(sample.Y)
}

const logEps = 1e-7

func Softmax(logits []*engine.Value) []*engine.Value {
	maxVal := logits[0].Data
	for _, l := range logits[1:] {
		if l.Data > maxVal {
			maxVal = l.Data
		}
	}

	shifted := make([]*engine.Value, len(logits))
	exps := make([]*engine.Value, len(logits))
	for i, l := range logits {
		shifted[i] = engine.Add(l, engine.Const(-maxVal))
		exps[i] = engine.Exp(shifted[i])
	}

	sum := engine.Const(0)
	for _, e := range exps {
		sum = engine.Add(sum, e)
	}

	probs := make([]*engine.Value, len(exps))
	for i, e := range exps {
		probs[i] = engine.Div(e, sum)
	}
	return probs
}

func oneHot(label int, numClasses int) map[int]float64 {
	v := make(map[int]float64, numClasses)
	v[label] = 1
	return v
}

func crossEntropyOneHot(probs []*engine.Value, oneHot map[int]float64) *engine.Value {
	terms := make([]*engine.Value, 0, len(oneHot))
	for i, p := range probs {
		if oneHot[i] == 0 {
			continue
		}
		logP := engine.Log(engine.Add(p, engine.Const(logEps)))
		terms = append(terms, engine.Mul(engine.Const(oneHot[i]), logP))
	}
	return engine.Neg(engine.Add(terms...))
}
