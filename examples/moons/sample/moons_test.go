package sample

import (
	"math/rand"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMakeMoons_shapeAndLabels(t *testing.T) {
	rng := rand.New(rand.NewSource(1337))
	data := MakeMoons(100, 0.1, rng)

	require.Len(t, data, 100)
	for _, s := range data {
		require.Len(t, s.X, 2)
		assert.True(t, s.Y == -1 || s.Y == 1)
	}
}

func TestEnsureData_generatesMoonsJSONL(t *testing.T) {
	require.NoError(t, EnsureData())
	_, err := os.Stat(dataPath("moons.jsonl"))
	require.NoError(t, err)

	data, err := ReadJSONL(dataPath("moons.jsonl"))
	require.NoError(t, err)
	require.Len(t, data, 100)
}
