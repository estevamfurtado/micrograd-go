package sample

import (
	"fmt"
	"image/color"
	"math"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/estevamfurtado/micrograd-go/engine"
	"github.com/estevamfurtado/micrograd-go/nn"
	"github.com/estevamfurtado/micrograd-go/nn/train"
)

type decisionGrid struct {
	xMin, yMin, step float64
	values           [][]float64
}

func (g *decisionGrid) Dims() (c, r int) {
	return len(g.values[0]), len(g.values)
}

func (g *decisionGrid) Z(c, r int) float64 {
	return g.values[r][c]
}

func (g *decisionGrid) X(c int) float64 {
	return g.xMin + float64(c)*g.step
}

func (g *decisionGrid) Y(r int) float64 {
	return g.yMin + float64(r)*g.step
}

func dataBounds(data train.Samples) (xMin, xMax, yMin, yMax float64) {
	xMin, xMax = data[0].X[0], data[0].X[0]
	yMin, yMax = data[0].X[1], data[0].X[1]
	for _, s := range data[1:] {
		xMin = math.Min(xMin, s.X[0])
		xMax = math.Max(xMax, s.X[0])
		yMin = math.Min(yMin, s.X[1])
		yMax = math.Max(yMax, s.X[1])
	}
	return xMin, xMax, yMin, yMax
}

func buildDecisionGrid(model *nn.MLP, data train.Samples, step float64) *decisionGrid {
	xMin, xMax, yMin, yMax := dataBounds(data)
	xMin, xMax = xMin-1, xMax+1
	yMin, yMax = yMin-1, yMax+1

	xs := arange(xMin, xMax, step)
	ys := arange(yMin, yMax, step)

	values := make([][]float64, len(ys))
	for r, y := range ys {
		values[r] = make([]float64, len(xs))
		for c, x := range xs {
			inputs := []*engine.Value{
				engine.Const(x),
				engine.Const(y),
			}
			score := model.Calculate(inputs)[0]
			if score.Data > 0 {
				values[r][c] = 1
			}
		}
	}

	return &decisionGrid{xMin: xMin, yMin: yMin, step: step, values: values}
}

func arange(start, stop, step float64) []float64 {
	n := int(math.Ceil((stop - start) / step))
	if n <= 0 {
		return nil
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = start + float64(i)*step
	}
	return out
}

type twoColorPalette struct {
	colors []color.Color
}

func (p twoColorPalette) Colors() []color.Color {
	return p.colors
}

func PlotDecisionBoundary(model *nn.MLP, data train.Samples, path string) error {
	const step = 0.25

	grid := buildDecisionGrid(model, data, step)

	pal := twoColorPalette{colors: []color.Color{
		color.RGBA{R: 30, G: 144, B: 255, A: 204},
		color.RGBA{R: 255, G: 69, B: 0, A: 204},
	}}

	heatmap := plotter.NewHeatMap(grid, pal)
	heatmap.Rasterized = true

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

	negScatter, err := plotter.NewScatter(negative)
	if err != nil {
		return err
	}
	negScatter.GlyphStyle.Color = color.RGBA{R: 30, G: 144, B: 255, A: 255}
	negScatter.GlyphStyle.Radius = vg.Points(4)

	posScatter, err := plotter.NewScatter(positive)
	if err != nil {
		return err
	}
	posScatter.GlyphStyle.Color = color.RGBA{R: 255, G: 69, B: 0, A: 255}
	posScatter.GlyphStyle.Radius = vg.Points(4)

	p := plot.New()
	p.Title.Text = "decision boundary"
	p.X.Label.Text = "x0"
	p.Y.Label.Text = "x1"
	p.Add(heatmap, negScatter, posScatter)

	xMin, xMax, yMin, yMax := dataBounds(data)
	p.X.Min, p.X.Max = xMin-1, xMax+1
	p.Y.Min, p.Y.Max = yMin-1, yMax+1

	if err := p.Save(5*vg.Inch, 5*vg.Inch, path); err != nil {
		return err
	}

	fmt.Printf("saved %s\n", path)
	return nil
}

func PlotDecisionBoundaryOrExit(model *nn.MLP, data train.Samples) {
	path := "decision_boundary.png"
	if err := PlotDecisionBoundary(model, data, path); err != nil {
		fmt.Fprintf(os.Stderr, "decision boundary plot: %v\n", err)
		os.Exit(1)
	}
}
