package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValue(t *testing.T) {
	a, b := Const(4.0), Const(2.0)

	t.Run("Add", func(t *testing.T) {
		c := Add(a, b)
		assert.Equal(t, c.Data, 6.0)
	})

	t.Run("Mul", func(t *testing.T) {
		c := Mul(a, b)
		assert.Equal(t, c.Data, 8.0)
	})

	t.Run("Pow", func(t *testing.T) {
		c := Pow(b, 3)
		assert.Equal(t, c.Data, 8.0)
	})

	t.Run("Div", func(t *testing.T) {
		c := Div(a, b)
		assert.Equal(t, c.Data, 2.0)
	})

	t.Run("Neg", func(t *testing.T) {
		c := Neg(a)
		assert.Equal(t, c.Data, -4.0)
	})

	t.Run("ReLU", func(t *testing.T) {
		c := ReLU(a)
		assert.Equal(t, c.Data, 4.0)

		a.Data = -4.0
		c = ReLU(a)
		assert.Equal(t, c.Data, 0.0)
	})

	t.Run("Exp", func(t *testing.T) {
		a.Data = 0.0
		c := Exp(a)
		assert.InDelta(t, 1.0, c.Data, 1e-9)
	})

	t.Run("Log", func(t *testing.T) {
		a.Data = 1.0
		c := Log(a)
		assert.InDelta(t, 0.0, c.Data, 1e-9)
	})
}

func TestValue_Backward(t *testing.T) {
	i1, i2 := Const(4.0), Const(2.0)

	// n1
	w11, w12, b1 := Const(0.1), Const(0.2), Const(0.3)
	n1 := ReLU(Add(Mul(i1, w11), Mul(i2, w12), b1))
	assert.Equal(t, 1.1, n1.Data) // ReLU[ w11 * i1 + w12 * i2 + b1 ]

	// n2
	w21, w22, b2 := Const(-0.1), Const(0.2), Const(0.3)
	n2 := ReLU(Add(Mul(i1, w21), Mul(i2, w22), b2)) // Relu[ w21 * i1 + w22 * i2 + b2 ]
	assert.Equal(t, 0.3, n2.Data)

	// combine n1 and n2
	out := Add(n1, Mul(n2, Const(2.0)))              // n1 + 2 * n2
	assert.Equal(t, float32(1.7), float32(out.Data)) // precision issues

	out.Backward()
	// assert gradients
	assert.Equal(t, out.Grad, 1.0)
	assert.Equal(t, n1.Grad, out.Grad*1.0)
	assert.Equal(t, n2.Grad, out.Grad*2.0)

	// n1
	assert.Equal(t, b1.Grad, n1.Grad)
	assert.Equal(t, w11.Grad, n1.Grad*i1.Data)
	assert.Equal(t, w12.Grad, n1.Grad*i2.Data)

	// n2
	assert.Equal(t, b2.Grad, n2.Grad)
	assert.Equal(t, w21.Grad, i1.Data*n2.Grad)
	assert.Equal(t, w22.Grad, i2.Data*n2.Grad)

	// inputs — ∂out/∂i1 = w11·∂out/∂n1 + w21·∂out/∂n2
	assert.Equal(t, w11.Data*n1.Grad+w21.Data*n2.Grad, i1.Grad)
	assert.Equal(t, w12.Data*n1.Grad+w22.Data*n2.Grad, i2.Grad)
}
