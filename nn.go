package main

import (
	"math"
	"math/rand"
)

type ActivationType int

const (
	ActivationLeakyReLU ActivationType = 0
	ActivationReLU      ActivationType = 1
	ActivationSigmoid   ActivationType = 2
	ActivationTanh      ActivationType = 3
)

func (a ActivationType) String() string {
	switch a {
	case ActivationLeakyReLU:
		return "LeakyReLU"
	case ActivationReLU:
		return "ReLU"
	case ActivationSigmoid:
		return "Sigmoid"
	case ActivationTanh:
		return "Tanh"
	default:
		return "Unknown"
	}
}

type OutputType int

const (
	OutputSoftmax OutputType = 0
	OutputSigmoid OutputType = 1
	OutputLinear  OutputType = 2
)

func (o OutputType) String() string {
	switch o {
	case OutputSoftmax:
		return "Softmax"
	case OutputSigmoid:
		return "Sigmoid"
	case OutputLinear:
		return "Linear"
	default:
		return "Unknown"
	}
}

type CostType int

const (
	CostCrossEntropy CostType = 0
	CostMSE          CostType = 1
)

func (c CostType) String() string {
	switch c {
	case CostCrossEntropy:
		return "Cross-Entropy"
	case CostMSE:
		return "MSE"
	default:
		return "Unknown"
	}
}

type DataPoint struct {
	inputs  []float32
	outputs []float32
}

type Layer struct {
	numNodesIn           int
	numNodesOut          int
	isOutput             bool
	activationType       ActivationType
	outputType           OutputType
	costType             CostType
	momentum             float32
	weights              [][]float32
	weightsBacking       []float32
	costGradientW        [][]float32
	costGradientWBacking []float32
	velocityW            [][]float32
	velocityWBacking     []float32
	inputs               []float32
	biases               []float32
	costGradientB        []float32
	velocityB            []float32
	weightedInputs       []float32
	activations          []float32
	nodeValues           []float32
}

func NewLayer(numNodesIn int, numNodesOut int, isOutput bool) *Layer {
	rows, cols := numNodesIn, numNodesOut
	weightsBacking := make([]float32, rows*cols)
	weightMatrix := make([][]float32, rows)
	for i := range weightMatrix {
		weightMatrix[i] = weightsBacking[i*cols : (i+1)*cols]
	}

	gradBacking := make([]float32, rows*cols)
	gradMatrix := make([][]float32, rows)
	for i := range gradMatrix {
		gradMatrix[i] = gradBacking[i*cols : (i+1)*cols]
	}

	velBacking := make([]float32, rows*cols)
	velMatrix := make([][]float32, rows)
	for i := range velMatrix {
		velMatrix[i] = velBacking[i*cols : (i+1)*cols]
	}

	l := &Layer{
		numNodesIn:           numNodesIn,
		numNodesOut:          numNodesOut,
		isOutput:             isOutput,
		activationType:       ActivationLeakyReLU,
		outputType:           OutputSoftmax,
		costType:             CostCrossEntropy,
		momentum:             0.0,
		weights:              weightMatrix,
		weightsBacking:       weightsBacking,
		costGradientW:        gradMatrix,
		costGradientWBacking: gradBacking,
		velocityW:            velMatrix,
		velocityWBacking:     velBacking,
		inputs:               make([]float32, numNodesIn),
		biases:               make([]float32, numNodesOut),
		costGradientB:        make([]float32, numNodesOut),
		velocityB:            make([]float32, numNodesOut),
		weightedInputs:       make([]float32, numNodesOut),
		activations:          make([]float32, numNodesOut),
		nodeValues:           make([]float32, numNodesOut),
	}
	InitializeRandomWeights(l)
	return l
}

func CalculateLayerOutputs(l *Layer, inputs []float32) []float32 {
	const maxWeightedInput float32 = 80.0
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		l.weightedInputs[nodeOut] = l.biases[nodeOut]
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			l.weightedInputs[nodeOut] += inputs[nodeIn] * l.weights[nodeIn][nodeOut]
		}
		// Bound weighted inputs to prevent float32 overflow (+Inf) in deep/wide networks
		if l.weightedInputs[nodeOut] > maxWeightedInput {
			l.weightedInputs[nodeOut] = maxWeightedInput
		} else if l.weightedInputs[nodeOut] < -maxWeightedInput {
			l.weightedInputs[nodeOut] = -maxWeightedInput
		}
	}

	if l.isOutput {
		switch l.outputType {
		case OutputSoftmax:
			// Numerically stable Softmax activation
			maxZ := l.weightedInputs[0]
			for i := 1; i < l.numNodesOut; i++ {
				if l.weightedInputs[i] > maxZ {
					maxZ = l.weightedInputs[i]
				}
			}
			sumExp := float32(0.0)
			for i := 0; i < l.numNodesOut; i++ {
				expVal := float32(math.Exp(float64(l.weightedInputs[i] - maxZ)))
				l.activations[i] = expVal
				sumExp += expVal
			}
			if sumExp > 0 {
				invSum := 1.0 / sumExp
				for i := 0; i < l.numNodesOut; i++ {
					l.activations[i] *= invSum
				}
			} else {
				invN := 1.0 / float32(l.numNodesOut)
				for i := 0; i < l.numNodesOut; i++ {
					l.activations[i] = invN
				}
			}
		case OutputSigmoid:
			for i := 0; i < l.numNodesOut; i++ {
				l.activations[i] = ActivationFunctionSigmoid(l.weightedInputs[i])
			}
		case OutputLinear:
			for i := 0; i < l.numNodesOut; i++ {
				l.activations[i] = l.weightedInputs[i]
			}
		}
	} else {
		switch l.activationType {
		case ActivationLeakyReLU:
			for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut++ {
				l.activations[nodeOut] = ActivationFunctionLeakyRELU(l.weightedInputs[nodeOut])
			}
		case ActivationReLU:
			for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut++ {
				l.activations[nodeOut] = ActivationFunctionRELU(l.weightedInputs[nodeOut])
			}
		case ActivationSigmoid:
			for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut++ {
				l.activations[nodeOut] = ActivationFunctionSigmoid(l.weightedInputs[nodeOut])
			}
		case ActivationTanh:
			for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut++ {
				l.activations[nodeOut] = ActivationFunctionTanh(l.weightedInputs[nodeOut])
			}
		}
	}
	l.inputs = inputs

	return l.activations
}

func CalculateOutputLayerNodeValues(l *Layer, expectedOutputs []float32) []float32 {
	const maxNodeValue float32 = 50.0

	switch l.costType {
	case CostCrossEntropy:
		if l.outputType == OutputLinear {
			const eps = 1e-7
			for i := 0; i < l.numNodesOut; i++ {
				a := l.activations[i]
				if a < eps {
					a = eps
				}
				val := -expectedOutputs[i] / a
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					val = 0.0
				} else if val > maxNodeValue {
					val = maxNodeValue
				} else if val < -maxNodeValue {
					val = -maxNodeValue
				}
				l.nodeValues[i] = val
			}
		} else {
			// Softmax or Sigmoid with Cross-Entropy simplifies to (a - y)
			for i := 0; i < l.numNodesOut; i++ {
				diff := l.activations[i] - expectedOutputs[i]
				if math.IsNaN(float64(diff)) || math.IsInf(float64(diff), 0) {
					diff = 0.0
				}
				l.nodeValues[i] = diff
			}
		}

	case CostMSE:
		switch l.outputType {
		case OutputLinear:
			for i := 0; i < l.numNodesOut; i++ {
				diff := l.activations[i] - expectedOutputs[i]
				if math.IsNaN(float64(diff)) || math.IsInf(float64(diff), 0) {
					diff = 0.0
				}
				l.nodeValues[i] = diff
			}
		case OutputSigmoid:
			for i := 0; i < l.numNodesOut; i++ {
				a := l.activations[i]
				diff := (a - expectedOutputs[i]) * a * (1.0 - a)
				if math.IsNaN(float64(diff)) || math.IsInf(float64(diff), 0) {
					diff = 0.0
				}
				l.nodeValues[i] = diff
			}
		case OutputSoftmax:
			sumErrA := float32(0.0)
			for k := 0; k < l.numNodesOut; k++ {
				sumErrA += (l.activations[k] - expectedOutputs[k]) * l.activations[k]
			}
			for i := 0; i < l.numNodesOut; i++ {
				a := l.activations[i]
				val := a * ((a - expectedOutputs[i]) - sumErrA)
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					val = 0.0
				}
				l.nodeValues[i] = val
			}
		}
	}

	return l.nodeValues
}

func ApplyGradients(l *Layer, lr float32) {
	const maxGradStep float32 = 1.0
	const maxWeight float32 = 10.0
	mom := l.momentum

	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		gradB := l.costGradientB[nodeOut]
		if math.IsNaN(float64(gradB)) || math.IsInf(float64(gradB), 0) {
			gradB = 0
		}
		l.velocityB[nodeOut] = mom*l.velocityB[nodeOut] + lr*gradB
		stepB := l.velocityB[nodeOut]
		if stepB > maxGradStep {
			stepB = maxGradStep
		} else if stepB < -maxGradStep {
			stepB = -maxGradStep
		}
		newB := l.biases[nodeOut] - stepB
		if newB > maxWeight {
			newB = maxWeight
		} else if newB < -maxWeight {
			newB = -maxWeight
		}
		l.biases[nodeOut] = newB

		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			gradW := l.costGradientW[nodeIn][nodeOut]
			if math.IsNaN(float64(gradW)) || math.IsInf(float64(gradW), 0) {
				gradW = 0
			}
			l.velocityW[nodeIn][nodeOut] = mom*l.velocityW[nodeIn][nodeOut] + lr*gradW
			stepW := l.velocityW[nodeIn][nodeOut]
			if stepW > maxGradStep {
				stepW = maxGradStep
			} else if stepW < -maxGradStep {
				stepW = -maxGradStep
			}
			newW := l.weights[nodeIn][nodeOut] - stepW
			if newW > maxWeight {
				newW = maxWeight
			} else if newW < -maxWeight {
				newW = -maxWeight
			}
			l.weights[nodeIn][nodeOut] = newW
		}
	}
}

func InitializeRandomWeights(l *Layer) {
	for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut += 1 {
		for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn += 1 {
			rv := rand.Float32()*2.0 - 1.0
			l.weights[nodeIn][nodeOut] = rv / float32(math.Sqrt(float64(l.numNodesIn)))
		}
	}
}

func ActivationFunctionLeakyRELU(weightedInput float32) float32 {
	if weightedInput > 0 {
		return weightedInput
	}
	return 0.01 * weightedInput
}

func DerivativeActivationFunctionLeakyRELU(weightedInput float32) float32 {
	if weightedInput > 0 {
		return 1.0
	}
	return 0.01
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

func ActivationFunctionSigmoid(weightedInput float32) float32 {
	if weightedInput > 40.0 {
		return 1.0
	} else if weightedInput < -40.0 {
		return 0.0
	}
	return float32(1.0 / (1.0 + math.Exp(-float64(weightedInput))))
}

func DerivativeActivationFunctionSigmoid(activation float32) float32 {
	return activation * (1.0 - activation)
}

func ActivationFunctionTanh(weightedInput float32) float32 {
	if weightedInput > 20.0 {
		return 1.0
	} else if weightedInput < -20.0 {
		return -1.0
	}
	return float32(math.Tanh(float64(weightedInput)))
}

func DerivativeActivationFunctionTanh(activation float32) float32 {
	return 1.0 - (activation * activation)
}

type NeuralNetwork struct {
	layers         []*Layer
	activationType ActivationType
	outputType     OutputType
	costType       CostType
	momentum       float32
}

func NewNeuralNetwork(layerSizes []int) *NeuralNetwork {
	return NewConfiguredNeuralNetwork(layerSizes, ActivationLeakyReLU, OutputSoftmax, CostCrossEntropy, 0.0)
}

func NewConfiguredNeuralNetwork(layerSizes []int, act ActivationType, out OutputType, cost CostType, momentum float32) *NeuralNetwork {
	layers := make([]*Layer, len(layerSizes)-1)
	for i := 0; i < len(layers); i += 1 {
		isOutput := (i == len(layers) - 1)
		layers[i] = NewLayer(layerSizes[i], layerSizes[i+1], isOutput)
		layers[i].activationType = act
		layers[i].outputType = out
		layers[i].costType = cost
		layers[i].momentum = momentum
	}
	return &NeuralNetwork{
		layers:         layers,
		activationType: act,
		outputType:     out,
		costType:       cost,
		momentum:       momentum,
	}
}

func (n *NeuralNetwork) SetActivationType(act ActivationType) {
	n.activationType = act
	for _, l := range n.layers {
		l.activationType = act
	}
}

func (n *NeuralNetwork) SetOutputType(out OutputType) {
	n.outputType = out
	for _, l := range n.layers {
		l.outputType = out
	}
}

func (n *NeuralNetwork) SetCostType(cost CostType) {
	n.costType = cost
	for _, l := range n.layers {
		l.costType = cost
	}
}

func (n *NeuralNetwork) SetMomentum(momentum float32) {
	n.momentum = momentum
	for _, l := range n.layers {
		l.momentum = momentum
	}
}

func (n *NeuralNetwork) ClearVelocities() {
	for _, l := range n.layers {
		clear(l.velocityWBacking)
		clear(l.velocityB)
	}
}

func CalculateOutputs(n *NeuralNetwork, inputs []float32) []float32 {
	for i := 0; i < len(n.layers); i += 1 {
		inputs = CalculateLayerOutputs(n.layers[i], inputs)
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
	const eps = 1e-7

	switch n.costType {
	case CostCrossEntropy:
		for i := 0; i < len(outputs); i += 1 {
			p := outputs[i]
			if p < eps {
				p = eps
			} else if p > 1.0 {
				p = 1.0
			}
			cost -= dataPoint.outputs[i] * float32(math.Log(float64(p)))
		}
	case CostMSE:
		for i := 0; i < len(outputs); i += 1 {
			diff := outputs[i] - dataPoint.outputs[i]
			cost += 0.5 * diff * diff
		}
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
	for i := 0; i < len(data); i += 1 {
		UpdateAllGradients(n, data[i])
	}
	ApplyAllGradients(n, lr/float32(len(data)))
	ClearAllGradients(n)
}

func ClearAllGradients(n *NeuralNetwork) {
	for i := range n.layers {
		l := n.layers[i]
		if len(l.costGradientWBacking) > 0 {
			clear(l.costGradientWBacking)
		} else {
			for r := range l.costGradientW {
				clear(l.costGradientW[r])
			}
		}
		clear(l.costGradientB)
	}
}

func ApplyAllGradients(n *NeuralNetwork, lr float32) {
	for i := 0; i < len(n.layers); i += 1 {
		ApplyGradients(n.layers[i], lr)
	}
}

func UpdateAllGradients(n *NeuralNetwork, dataPoint DataPoint) {
	CalculateOutputs(n, dataPoint.inputs)
	numLayers := len(n.layers)
	outputLayer := n.layers[numLayers-1]
	nodeValues := CalculateOutputLayerNodeValues(outputLayer, dataPoint.outputs)
	UpdateGradients(outputLayer, nodeValues)
	for hiddenIndex := numLayers - 2; hiddenIndex >= 0; hiddenIndex -= 1 {
		hiddenLayer := n.layers[hiddenIndex]
		nodeValues = CalculateHiddenLayerNodeValues(hiddenLayer, n.layers[hiddenIndex+1], nodeValues)
		UpdateGradients(hiddenLayer, nodeValues)
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
	const maxNodeValue float32 = 50.0
	for newNodeIndex := 0; newNodeIndex < nl.numNodesOut; newNodeIndex += 1 {
		newNodeValue := float32(0.0)
		for oldNodeIndex := 0; oldNodeIndex < len(oldNodeValues); oldNodeIndex += 1 {
			weightedInputDerivative := ol.weights[newNodeIndex][oldNodeIndex]
			newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex]
		}
		var actDeriv float32
		switch nl.activationType {
		case ActivationLeakyReLU:
			actDeriv = DerivativeActivationFunctionLeakyRELU(nl.weightedInputs[newNodeIndex])
		case ActivationReLU:
			actDeriv = DerivativeActivationFunctionRELU(nl.weightedInputs[newNodeIndex])
		case ActivationSigmoid:
			actDeriv = DerivativeActivationFunctionSigmoid(nl.activations[newNodeIndex])
		case ActivationTanh:
			actDeriv = DerivativeActivationFunctionTanh(nl.activations[newNodeIndex])
		}
		newNodeValue *= actDeriv
		if math.IsNaN(float64(newNodeValue)) || math.IsInf(float64(newNodeValue), 0) {
			newNodeValue = 0.0
		} else if newNodeValue > maxNodeValue {
			newNodeValue = maxNodeValue
		} else if newNodeValue < -maxNodeValue {
			newNodeValue = -maxNodeValue
		}
		nl.nodeValues[newNodeIndex] = newNodeValue
	}
	return nl.nodeValues
}

func (l *Layer) GetWeight(nodeIn, nodeOut int) float32 {
	if len(l.weightsBacking) > 0 {
		return l.weightsBacking[nodeIn*l.numNodesOut+nodeOut]
	}
	return l.weights[nodeIn][nodeOut]
}

func (l *Layer) SetWeight(nodeIn, nodeOut int, val float32) {
	if len(l.weightsBacking) > 0 {
		l.weightsBacking[nodeIn*l.numNodesOut+nodeOut] = val
	} else {
		l.weights[nodeIn][nodeOut] = val
	}
}

func (l *Layer) GetWeightGradient(nodeIn, nodeOut int) float32 {
	if len(l.costGradientWBacking) > 0 {
		return l.costGradientWBacking[nodeIn*l.numNodesOut+nodeOut]
	}
	return l.costGradientW[nodeIn][nodeOut]
}

func CalculateAccuracy(n *NeuralNetwork, data []DataPoint) float32 {
	if len(data) == 0 {
		return 0.0
	}
	correct := 0
	for i := range data {
		pred := Classify(n, data[i].inputs)
		target := IndexOfMaxValue(data[i].outputs)
		if pred == target {
			correct++
		}
	}
	return (float32(correct) / float32(len(data))) * 100.0
}

func ShuffleDataPoints(data []DataPoint) {
	for i := len(data) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
}

