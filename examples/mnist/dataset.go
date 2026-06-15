package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
)

const (
	numPixels  = 28 * 28
	numClasses = 10
)

// Sample is one MNIST image: normalized pixels and integer label 0-9.
type Sample struct {
	Label int
	X     [numPixels]float64
}

type Samples []Sample

// OneHot encodes a digit as a length-10 vector with a 1 at the label index.
func OneHot(label int) [numClasses]float64 {
	var v [numClasses]float64
	v[label] = 1
	return v
}

// LoadCSV reads pjreddie MNIST CSV: label,784 pixel columns (0-255).
// Pixels are normalized to [0, 1]. limit caps rows (0 = all).
func LoadCSV(path string, limit int) (Samples, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	samples := Samples{}

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
		if len(record) != numPixels+1 {
			return nil, fmt.Errorf("expected %d columns, got %d", numPixels+1, len(record))
		}

		label, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, fmt.Errorf("label: %w", err)
		}
		if label < 0 || label >= numClasses {
			return nil, fmt.Errorf("label out of range: %d", label)
		}

		var s Sample
		s.Label = label
		for i := 0; i < numPixels; i++ {
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

func (s Samples) Shuffle(rng *rand.Rand) {
	rng.Shuffle(len(s), func(i, j int) {
		s[i], s[j] = s[j], s[i]
	})
}
