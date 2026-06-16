package train

import "math/rand"

type Sample struct {
	X []float64
	Y float64
}

type Samples []Sample

func (s Samples) Shuffle(rng *rand.Rand) {
	rng.Shuffle(len(s), func(i, j int) {
		s[i], s[j] = s[j], s[i]
	})
}
