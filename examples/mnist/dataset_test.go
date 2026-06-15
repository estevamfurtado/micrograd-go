package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadCSV_normalizesPixels(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "one.csv")

	pixels := make([]string, numPixels)
	pixels[0] = "255"
	for i := 1; i < numPixels; i++ {
		pixels[i] = "0"
	}
	row := "7," + strings.Join(pixels, ",") + "\n"
	require.NoError(t, os.WriteFile(path, []byte(row), 0o644))

	samples, err := LoadCSV(path, 0)
	require.NoError(t, err)
	require.Len(t, samples, 1)
	assert.Equal(t, 7, samples[0].Label)
	assert.InDelta(t, 1.0, samples[0].X[0], 1e-9)
	assert.InDelta(t, 0.0, samples[0].X[1], 1e-9)
}

func TestOneHot(t *testing.T) {
	assert.Equal(t, [numClasses]float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, OneHot(3))
}
