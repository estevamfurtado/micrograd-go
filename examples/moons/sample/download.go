package sample

import (
	"fmt"
	"math/rand"
	"os"
)

const (
	defaultNSamples = 100
	defaultNoise    = 0.1
)

// EnsureData generates moons.jsonl on first run.
func EnsureData() error {
	dir := DataDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	path := dataPath("moons.jsonl")
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	rng := rand.New(rand.NewSource(1337))
	data := MakeMoons(defaultNSamples, defaultNoise, rng)
	if err := WriteJSONL(path, data); err != nil {
		return err
	}
	fmt.Printf("generated %s (%d samples)\n", path, len(data))
	return nil
}
