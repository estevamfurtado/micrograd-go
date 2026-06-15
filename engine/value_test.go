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
}

func TestValue_Backward(t *testing.T) {
	i1, i2 := Const(4.0), Const(2.0)

	// n1
	w11, w12, b1 := Const(0.1), Const(0.2), Const(0.3)
	n1 := ReLU(Add(Mul(i1, w11), Mul(i2, w12), b1))
	assert.Equal(t, n1.Data, 1.1)

	// n2
	w21, w22, b2 := Const(-0.1), Const(0.2), Const(0.3)
	n2 := ReLU(Add(Mul(i1, w21), Mul(i2, w22), b2))
	assert.Equal(t, n2.Data, 0.3)

	// combine n1 and n2
	o3 := Add(n1, n2)
	assert.Equal(t, float32(o3.Data), float32(1.4)) // precision issues

	o3.Backward()
}
