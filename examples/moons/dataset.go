package main

import (
	"fmt"
	"image/color"
	"math/rand"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Default values:
// nSamples = 100
// noise = 0.1
func GenerateDataset(nSamples int, noise float64) {
	rng := rand.New(rand.NewSource(1337))
	data := MakeMoons(nSamples, noise, rng)

	jsonlPath := "examples/moons/moons.jsonl"
	if err := WriteJSONL(jsonlPath, data); err != nil {
		fmt.Fprintf(os.Stderr, "jsonl export: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("saved %s (%d samples)\n", jsonlPath, len(data))

	positive := plotter.XYs{}
	negative := plotter.XYs{}
	for _, s := range data {
		pt := plotter.XY{X: s.X[0], Y: s.X[1]}
		if s.Y > 0 {
			positive = append(positive, pt)
		} else {
			negative = append(negative, pt)
		}
	}

	p := plot.New()
	p.Title.Text = "make_moons (n=100, noise=0.1)"
	p.X.Label.Text = "x0"
	p.Y.Label.Text = "x1"

	negScatter, err := plotter.NewScatter(negative)
	if err != nil {
		panic(err)
	}
	negScatter.GlyphStyle.Color = color.RGBA{R: 30, G: 144, B: 255, A: 255}

	posScatter, err := plotter.NewScatter(positive)
	if err != nil {
		panic(err)
	}
	posScatter.GlyphStyle.Color = color.RGBA{R: 255, G: 69, B: 0, A: 255}

	p.Add(negScatter, posScatter)
	p.Legend.Add("y = -1", negScatter)
	p.Legend.Add("y = +1", posScatter)
	p.Legend.Top = true

	out := "moons.png"
	if err := p.Save(5*vg.Inch, 5*vg.Inch, out); err != nil {
		fmt.Fprintf(os.Stderr, "plot save: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("saved %s (%d samples)\n", out, len(data))
	fmt.Printf("labels: y in {-1, +1} — same as y = y*2 - 1 in Python\n")
}

func loadMoons() Samples {
	jsonlPath := "examples/moons/moons.jsonl"
	data, err := ReadJSONL(jsonlPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "jsonl load: %v\n", err)
		os.Exit(1)
	}

	return data
}
