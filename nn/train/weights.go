package train

import (
	"math"
	"math/rand"

	"github.com/estevamfurtado/micrograd-go/nn"
)

// HeInit is uniform He/Kaiming init for ReLU layers:
// w ~ U(-sqrt(2/fanIn), +sqrt(2/fanIn)), b = 0.
var HeInit = nn.ParamsFactory{
	BiasGenerator: func() float64 { return 0 },
	WeightGenerator: func(fanIn int) float64 {
		bound := math.Sqrt(2.0 / float64(fanIn))
		return (rand.Float64()*2 - 1) * bound
	},
}

// XavierInit is uniform Glorot init for linear layers:
// w ~ U(-sqrt(1/fanIn), +sqrt(1/fanIn)), b = 0.
var XavierInit = nn.ParamsFactory{
	BiasGenerator: func() float64 { return 0 },
	WeightGenerator: func(fanIn int) float64 {
		bound := math.Sqrt(1.0 / float64(fanIn))
		return (rand.Float64()*2 - 1) * bound
	},
}

// RandomInit draws weights and bias uniformly from [-1, 1], ignoring fanIn.
var RandomInit = nn.ParamsFactory{
	BiasGenerator: func() float64 { return rand.Float64()*2 - 1 },
	WeightGenerator: func(fanIn int) float64 {
		return rand.Float64()*2 - 1
	},
}
