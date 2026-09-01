package main

import "math"

type Layer struct {
	numNodesIn  int
	numNodesOut int
	weights     [][]float32
	biases      []float32
}

func NewLayer(numNodesIn int, numNodesOut int) *Layer {
	rows, cols := numNodesIn, numNodesOut
	matrix := make([][]float32, rows)
	for i := range matrix {
		matrix[i] = make([]float32, cols)
	}
	return &Layer{
		numNodesIn:  numNodesIn,
		numNodesOut: numNodesOut,
		weights:     matrix,
		biases:      make([]float32, numNodesOut),
	}
}

func CalculateLayerOutputs(l *Layer, inputs []float32) []float32 {
	weightedInputs := make([]float32, l.numNodesOut)
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		weightedInput := l.biases[nodeOut]
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			weightedInput += inputs[nodeIn] * l.weights[nodeIn][nodeOut]
		}
		weightedInputs[nodeOut] = ActivationFunctionSig(weightedInput)
	}
	return weightedInputs
}
func ActivationFunctionRELU(weightedInput float32) float32 {
	if weightedInput > 0 {
		return weightedInput
	} else {
		return 0.0
	}
}
func ActivationFunctionSig(weightedInput float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(weightedInput))))
}

type NeuralNetwork struct {
	layers []Layer
}

func NewNeuralNetwork(layerSizes []int) *NeuralNetwork {
	layers := make([]Layer, len(layerSizes)-1)
	for i := 0; i < len(layers); i += 1 {
		layers[i] = *NewLayer(layerSizes[i], layerSizes[i+1])
	}
	return &NeuralNetwork{
		layers: layers,
	}
}

func CalculateOutputs(n *NeuralNetwork, inputs []float32) []float32 {
	for i := 0; i < len(n.layers); i += 1 {
		inputs = CalculateLayerOutputs(&n.layers[i], inputs)
	}
	return inputs
}

func Classify(n *NeuralNetwork, inputs []float32) int {
	outputs := CalculateOutputs(n, inputs)
	return IndexOfMaxValue(outputs)
}

func IndexOfMaxValue(outputs []float32) int {
	res := 0
	for i := 0; i < len(outputs); i += 1 {
		if outputs[res] < outputs[i] {
			res = i
		}
	}
	return res
}
