package sample

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/estevamfurtado/micrograd-go/nn/train"
)

const (
	NumPixels  = 28 * 28
	NumClasses = 10
)

// LoadCSV reads pjreddie MNIST CSV: label,784 pixel columns (0-255).
// Pixels are normalized to [0, 1]. limit caps rows (0 = all).
func LoadCSV(path string, limit int) (train.Samples, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	samples := train.Samples{}

	for {
		if limit > 0 && len(samples) >= limit {
			break
		}

		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		if len(record) != NumPixels+1 {
			return nil, fmt.Errorf("expected %d columns, got %d", NumPixels+1, len(record))
		}

		label, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, fmt.Errorf("label: %w", err)
		}
		if label < 0 || label >= NumClasses {
			return nil, fmt.Errorf("label out of range: %d", label)
		}

		s := train.Sample{
			Y: float64(label),
			X: make([]float64, NumPixels),
		}
		for i := 0; i < NumPixels; i++ {
			pix, err := strconv.Atoi(record[i+1])
			if err != nil {
				return nil, fmt.Errorf("pixel %d: %w", i, err)
			}
			s.X[i] = float64(pix) / 255.0
		}
		samples = append(samples, s)
	}

	return samples, nil
}
