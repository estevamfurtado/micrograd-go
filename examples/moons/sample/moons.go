package sample

import (
	"math"
	"math/rand"

	"github.com/estevamfurtado/micrograd-go/nn/train"
)

// MakeMoons generates the same toy dataset as sklearn.datasets.make_moons.
func MakeMoons(nSamples int, noise float64, rng *rand.Rand) train.Samples {
	half := nSamples / 2
	samples := make(train.Samples, nSamples)

	for i := 0; i < half; i++ {
		t := math.Pi * float64(i) / float64(half-1)

		x := math.Cos(t) + rng.NormFloat64()*noise
		y := math.Sin(t) + rng.NormFloat64()*noise
		samples[i] = train.Sample{X: []float64{x, y}, Y: -1}

		x = 1 - math.Cos(t) + rng.NormFloat64()*noise
		y = 0.5 - math.Sin(t) + rng.NormFloat64()*noise
		samples[i+half] = train.Sample{X: []float64{x, y}, Y: 1}
	}

	return samples
}
