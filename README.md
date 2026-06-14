# micrograd-go

Reimplementação do [micrograd](https://github.com/karpathy/micrograd) do Andrej Karpathy em Go — projeto de estudo para reforçar autograd, backpropagation e redes neurais do zero.

> **Status:** scaffold inicial. Nada implementado ainda.

## Referências

- Repositório original: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- Vídeo-aula: [micrograd explained - backpropagation and training neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)

## O que é o micrograd?

Uma engine de autograd escalar (~100 linhas) + uma biblioteca de redes neurais mínima (~50 linhas) com API estilo PyTorch. Cada operação é um escalar; o grafo computacional é construído dinamicamente e o gradiente flui via backpropagation em modo reverso.

## Estrutura do projeto

```
micrograd-go/
├── engine/     # autograd — port de engine.py (Value, operadores, backward)
├── nn/         # redes neurais — port de nn.py (Module, Neuron, Layer, MLP)
└── examples/   # demos de treino (futuro)
```

## Roadmap de implementação

1. **engine** — struct `Value`, operadores aritméticos, `Tanh`, `ReLU`, `Pow`, `Backward()`
2. **nn** — `Module`, `Neuron`, `Layer`, `MLP` e coleta de parâmetros
3. **examples** — treinar um MLP em dataset toy (equivalente ao `demo.ipynb`)
4. **testes** — gradient checking com diferenças finitas

## Desenvolvimento

```bash
go build ./...
go test ./...
```

## Licença

MIT — mesmo espírito educacional do projeto original.
