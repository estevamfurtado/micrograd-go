package datasets

import (
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONL_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1337))
	original := MakeMoons(100, 0.1, rng)

	path := filepath.Join(t.TempDir(), "moons.jsonl")
	require.NoError(t, WriteJSONL(path, original))

	loaded, err := ReadJSONL(path)
	require.NoError(t, err)
	assert.Equal(t, original, loaded)
}
