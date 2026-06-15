package engine

import (
	"math"
)

type Op string

const (
	OpAdd   Op = "+"
	OpMul   Op = "x"
	OpPow   Op = "**"
	OpReLU  Op = "relu"
	OpExp   Op = "exp"
	OpLog   Op = "log"
	OpConst Op = "const"
)

type Value struct {
	Data     float64
	Op       Op
	Parents  []*Value
	Grad     float64
	backward func()
}

func New(data float64, op Op, parents []*Value) *Value {
	result := &Value{Data: data, Op: op, Parents: parents, backward: func() {}}
	return result
}

func Const(data float64) *Value {
	out := New(data, OpConst, []*Value{})
	return out
}

func (this *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, parent := range v.Parents {
				buildTopo(parent)
			}
			topo = append(topo, v)
		}
	}

	buildTopo(this)

	// reset gradient - is this correct?
	for _, v := range topo {
		v.Grad = 0
	}

	// set gradient of this to 1
	this.Grad = 1

	// apply chain rule (leaves may have nil backward)
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

func Add(vs ...*Value) *Value {
	result := 0.0
	for _, v := range vs {
		result += v.Data
	}

	out := New(result, OpAdd, vs)

	out.backward = func() {
		for _, v := range vs {
			v.Grad += 1 * out.Grad
		}
	}

	return out
}

func Mul(vs ...*Value) *Value {
	result := 1.0
	for _, v := range vs {
		result *= v.Data
	}

	out := New(result, OpMul, vs)

	out.backward = func() {
		for _, v := range vs {
			v.Grad += (result / v.Data) * out.Grad
		}
	}

	return out
}

func Pow(a *Value, b float64) *Value {
	result := math.Pow(a.Data, b)
	out := New(result, OpPow, []*Value{a})

	out.backward = func() {
		a.Grad += b * math.Pow(a.Data, b-1) * out.Grad
	}

	return out
}

func Div(a, b *Value) *Value {
	return Mul(a, Pow(b, -1))
}

func Neg(a *Value) *Value {
	return Mul(a, Const(-1))
}

func ReLU(a *Value) *Value {
	var result float64
	if a.Data > 0 {
		result = a.Data
	}

	out := New(result, OpReLU, []*Value{a})
	out.backward = func() {
		if a.Data > 0 {
			a.Grad += 1 * out.Grad
		}
	}

	return out
}

func Exp(a *Value) *Value {
	result := math.Exp(a.Data)
	out := New(result, OpExp, []*Value{a})
	out.backward = func() {
		a.Grad += result * out.Grad
	}
	return out
}

func Log(a *Value) *Value {
	result := math.Log(a.Data)
	out := New(result, OpLog, []*Value{a})
	out.backward = func() {
		a.Grad += (1.0 / a.Data) * out.Grad
	}
	return out
}
