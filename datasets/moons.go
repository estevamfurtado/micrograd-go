package datasets

import (
	"math"
	"math/rand"
)

// Sample is one point in a 2D classification dataset.
type Sample struct {
	X [2]float64 `json:"x"`
	Y float64    `json:"y"` // -1 or +1
}

type Samples []Sample

// MakeMoons generates the same toy dataset as sklearn.datasets.make_moons.
func MakeMoons(nSamples int, noise float64, rng *rand.Rand) Samples {
	half := nSamples / 2
	samples := make(Samples, nSamples)

	for i := 0; i < half; i++ {
		t := math.Pi * float64(i) / float64(half-1)

		x := math.Cos(t) + rng.NormFloat64()*noise
		y := math.Sin(t) + rng.NormFloat64()*noise
		samples[i] = Sample{X: [2]float64{x, y}, Y: -1}

		x = 1 - math.Cos(t) + rng.NormFloat64()*noise
		y = 0.5 - math.Sin(t) + rng.NormFloat64()*noise
		samples[i+half] = Sample{X: [2]float64{x, y}, Y: 1}
	}

	return samples
}

func (s Samples) Shuffle() {
	rng := rand.New(rand.NewSource(1337))
	rng.Shuffle(len(s), func(i, j int) {
		s[i], s[j] = s[j], s[i]
	})
}
