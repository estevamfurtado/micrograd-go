package sample

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/estevamfurtado/micrograd-go/nn/train/loss"
)

func TestLoadCSV_normalizesPixels(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "one.csv")

	pixels := make([]string, NumPixels)
	pixels[0] = "255"
	for i := 1; i < NumPixels; i++ {
		pixels[i] = "0"
	}
	row := "7," + strings.Join(pixels, ",") + "\n"
	require.NoError(t, os.WriteFile(path, []byte(row), 0o644))

	samples, err := LoadCSV(path, 0)
	require.NoError(t, err)
	require.Len(t, samples, 1)
	assert.Equal(t, 7.0, samples[0].Y)
	assert.InDelta(t, 1.0, samples[0].X[0], 1e-9)
	assert.InDelta(t, 0.0, samples[0].X[1], 1e-9)
}

func TestLoadCSV_limit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rows.csv")

	var rows []string
	for label := 0; label < 5; label++ {
		pixels := make([]string, NumPixels)
		for i := range pixels {
			pixels[i] = "0"
		}
		rows = append(rows, strings.Join(append([]string{strconv.Itoa(label)}, pixels...), ","))
	}
	require.NoError(t, os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o644))

	samples, err := LoadCSV(path, 2)
	require.NoError(t, err)
	require.Len(t, samples, 2)
	assert.Equal(t, 0.0, samples[0].Y)
	assert.Equal(t, 1.0, samples[1].Y)
}

func TestOneHot(t *testing.T) {
	assert.Equal(t, []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, loss.OneHot(3, NumClasses))
}
