package main

import (
	"math"
	"math/rand"
)

type DataPoint struct {
	inputs  []float32
	outputs []float32
}

type Layer struct {
	numNodesIn     int
	numNodesOut    int
	weights        [][]float32
	costGradientW  [][]float32
	inputs         []float32
	biases         []float32
	costGradientB  []float32
	weightedInputs []float32
	activations    []float32
}

func NewLayer(numNodesIn int, numNodesOut int) *Layer {
	rows, cols := numNodesIn, numNodesOut
	weightMatrix := make([][]float32, rows)
	for i := range weightMatrix {
		weightMatrix[i] = make([]float32, cols)
	}
	gradMatrix := make([][]float32, rows)
	for i := range gradMatrix {
		gradMatrix[i] = make([]float32, cols)
	}
	l := &Layer{
		numNodesIn:     numNodesIn,
		numNodesOut:    numNodesOut,
		weights:        weightMatrix,
		costGradientW:  gradMatrix,
		inputs:         make([]float32, numNodesIn),
		biases:         make([]float32, numNodesOut),
		costGradientB:  make([]float32, numNodesOut),
		weightedInputs: make([]float32, numNodesOut),
		activations:    make([]float32, numNodesOut),
	}
	InitializeRandomWeights(l)
	return l
}

func CalculateLayerOutputs(l *Layer, inputs []float32) []float32 {
	// weightedInputs := make([]float32, l.numNodesOut)
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		l.weightedInputs[nodeOut] = l.biases[nodeOut]
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			l.weightedInputs[nodeOut] += inputs[nodeIn] * l.weights[nodeIn][nodeOut]
		}
		l.activations[nodeOut] = ActivationFunctionRELU(l.weightedInputs[nodeOut])
		// weightedInputs[nodeOut] = ActivationFunctionSig(weightedInput)
	}
	l.inputs = inputs
	// println("HERE")

	return l.activations
}

func CalculateOutputLayerNodeValues(l *Layer, expectedOutputs []float32) []float32 {
	nodeValues := make([]float32, len(expectedOutputs))
	for i := 0; i < len(nodeValues); i += 1 {
		cd := NodeCostDerivative(l.activations[i], expectedOutputs[i])
		ad := DerivativeActivationFunctionRELU(l.weightedInputs[i])
		nodeValues[i] = ad * cd
	}
	return nodeValues
}

func ApplyGradients(l *Layer, lr float32) {
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		l.biases[nodeOut] -= l.costGradientB[nodeOut] * lr
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			l.weights[nodeIn][nodeOut] -= l.costGradientW[nodeIn][nodeOut] * lr
		}
	}
}

func InitializeRandomWeights(l *Layer) {
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			rv := rand.Float32()*float32(WEIGHT_RANGE_MULTIPLIER) - float32(WEIGHT_RANGE_ADDER)
			l.weights[nodeIn][nodeOut] = rv / float32(math.Sqrt(float64(l.numNodesIn)))
		}
	}
}

func ActivationFunctionRELU(weightedInput float32) float32 {
	if weightedInput > 0 {
		return weightedInput
	} else {
		return 0.0
	}
}
func DerivativeActivationFunctionRELU(weightedInput float32) float32 {
	if weightedInput > 0 {
		return 1.0
	} else {
		return 0.0
	}
}
func ActivationFunctionSig(weightedInput float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(weightedInput))))
}

func DerivativeActivationFunctionSig(weightedInput float32) float32 {
	act := ActivationFunctionSig(weightedInput)
	return act * (1.0 - act)
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

func NodeCost(outputActivation, expectedOutput float32) float32 {
	error := outputActivation - expectedOutput
	return error * error
}

func NodeCostDerivative(outputActivation, expectedOutput float32) float32 {
	return 2.0 * (outputActivation - expectedOutput)
}

func Cost(n *NeuralNetwork, dataPoint DataPoint) float32 {
	outputs := CalculateOutputs(n, dataPoint.inputs)
	cost := float32(0.0)
	for i := 0; i < len(outputs); i += 1 {
		cost += NodeCost(outputs[i], dataPoint.outputs[i])
	}
	return cost
}
func NetworkCost(n *NeuralNetwork, data []DataPoint) float32 {
	totalCost := float32(0)
	for i := 0; i < len(data); i += 1 {
		totalCost += Cost(n, data[i])
	}
	return totalCost / float32(len(data))
}

func Learn(n *NeuralNetwork, data []DataPoint, lr float32) {
	// OLD LEGACY SLOW CODE
	// h := float32(0.001)
	// ogCost := NetworkCost(n, data)
	// for i := 0; i < len(n.layers); i += 1 {
	// 	for nodeIn := 0; nodeIn < n.layers[i].numNodesIn; nodeIn += 1 {
	// 		for nodeOut := 0; nodeOut < n.layers[i].numNodesOut; nodeOut += 1 {
	// 			n.layers[i].weights[nodeIn][nodeOut] += h
	// 			delCost := NetworkCost(n, data) - ogCost
	// 			n.layers[i].weights[nodeIn][nodeOut] -= h
	// 			n.layers[i].costGradientW[nodeIn][nodeOut] = delCost / h

	// 		}
	// 	}
	// 	for bi := 0; bi < len(n.layers[i].biases); bi += 1 {
	// 		n.layers[i].biases[bi] += h
	// 		delCost := NetworkCost(n, data) - ogCost
	// 		n.layers[i].biases[bi] -= h
	// 		n.layers[i].costGradientB[bi] = delCost / h
	// 	}
	// }

	// ApplyAllGradients(n, lr)
	for i := 0; i < len(data); i += 1 {
		UpdateAllGradients(n, data[i])
	}
	ApplyAllGradients(n, lr/float32(len(data)))
	ClearAllGradients(n)
}

func ClearAllGradients(n *NeuralNetwork) {
	for i := 0; i < len(n.layers); i += 1 {
		rows, cols := n.layers[i].numNodesIn, n.layers[i].numNodesOut
		gradMatrix := make([][]float32, rows)
		for i := range gradMatrix {
			gradMatrix[i] = make([]float32, cols)
		}
		n.layers[i].costGradientW = gradMatrix
		n.layers[i].costGradientB = make([]float32, n.layers[i].numNodesOut)
	}
}

func ApplyAllGradients(n *NeuralNetwork, lr float32) {
	for i := 0; i < len(n.layers); i += 1 {
		ApplyGradients(&n.layers[i], lr)
	}
}

func UpdateAllGradients(n *NeuralNetwork, dataPoint DataPoint) {
	CalculateOutputs(n, dataPoint.inputs)
	num_layers := len(n.layers)
	outputLayer := n.layers[num_layers-1]
	nodeValues := CalculateOutputLayerNodeValues(&outputLayer, dataPoint.outputs)
	UpdateGradients(&outputLayer, nodeValues)
	for hiddenIndex := num_layers - 2; hiddenIndex >= 0; hiddenIndex -= 1 {
		hiddenLayer := n.layers[hiddenIndex]
		nodeValues = CalculateHiddenLayerNodeValues(&hiddenLayer, &n.layers[hiddenIndex+1], nodeValues)
		UpdateGradients(&hiddenLayer, nodeValues)
	}
}

func UpdateGradients(l *Layer, nodeValues []float32) {
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			deriCostWrtWeight := l.inputs[nodeIn] * nodeValues[nodeOut]
			l.costGradientW[nodeIn][nodeOut] += deriCostWrtWeight
		}
		deriCostWrtBias := 1 * nodeValues[nodeOut]
		l.costGradientB[nodeOut] += deriCostWrtBias
	}
}

func CalculateHiddenLayerNodeValues(nl, ol *Layer, oldNodeValues []float32) []float32 {
	newNodeValues := make([]float32, nl.numNodesOut)
	for newNodeIndex := 0; newNodeIndex < len(newNodeValues); newNodeIndex += 1 {
		newNodeValue := float32(0.0)
		for oldNodeIndex := 0; oldNodeIndex < len(oldNodeValues); oldNodeIndex += 1 {
			weightedInputDerivative := ol.weights[newNodeIndex][oldNodeIndex]
			newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex]
		}
		newNodeValue *= DerivativeActivationFunctionRELU(nl.weightedInputs[newNodeIndex])
		newNodeValues[newNodeIndex] = newNodeValue
	}
	return newNodeValues
}
