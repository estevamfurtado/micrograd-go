package main

import (
	"github.com/estevamfurtado/micrograd-go/engine"
)

type CrossEntropyCalculator struct{}

func (l *CrossEntropyCalculator) Calculate(logits []*engine.Value, sample Sample) *engine.Value {
	probs := Softmax(logits)
	oneHot := OneHot(sample.Label)
	return CrossEntropyOneHot(probs, oneHot)
}

func (l *CrossEntropyCalculator) IsAccurate(logits []*engine.Value, sample Sample) bool {
	return argmax(logits) == sample.Label
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

func CrossEntropyOneHot(probs []*engine.Value, oneHot [numClasses]float64) *engine.Value {
	terms := make([]*engine.Value, 0, numClasses)
	for i, p := range probs {
		if oneHot[i] == 0 {
			continue
		}
		logP := engine.Log(engine.Add(p, engine.Const(logEps)))
		terms = append(terms, engine.Mul(engine.Const(oneHot[i]), logP))
	}
	return engine.Neg(engine.Add(terms...))
}

func argmax(logits []*engine.Value) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i].Data > logits[best].Data {
			best = i
		}
	}
	return best
}
