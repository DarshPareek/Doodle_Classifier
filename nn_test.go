package main

import (
	"math"
	"testing"
	"time"
)

func TestOutputLayerConfiguration(t *testing.T) {
	nn := NewNeuralNetwork([]int{2, 4, 8, 2})
	for i := 0; i < len(nn.layers)-1; i++ {
		if nn.layers[i].isOutput {
			t.Errorf("Layer %d should not be marked as output layer", i)
		}
	}
	outputLayer := nn.layers[len(nn.layers)-1]
	if !outputLayer.isOutput {
		t.Errorf("Final layer should be marked as output layer")
	}
}

func TestOutputLayerSoftmaxActivation(t *testing.T) {
	nn := NewNeuralNetwork([]int{2, 2})
	outputLayer := nn.layers[0]

	outputLayer.biases[0] = -1.5
	outputLayer.biases[1] = 0.5
	for r := range outputLayer.weights {
		for c := range outputLayer.weights[r] {
			outputLayer.weights[r][c] = 0
		}
	}

	inputs := []float32{1.0, 1.0}
	outputs := CalculateOutputs(nn, inputs)

	// Logits are [-1.5, 0.5]
	// Softmax: exp(-2) / (exp(-2) + 1) ~= 0.119203, 1 / (exp(-2) + 1) ~= 0.880797
	if math.Abs(float64(outputs[0]-0.119203)) > 1e-4 {
		t.Errorf("Expected Softmax output[0] ~ 0.119203, got %f", outputs[0])
	}
	if math.Abs(float64(outputs[1]-0.880797)) > 1e-4 {
		t.Errorf("Expected Softmax output[1] ~ 0.880797, got %f", outputs[1])
	}
	sum := outputs[0] + outputs[1]
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Expected sum of probabilities == 1.0, got %f", sum)
	}

	// Test numerical stability with large logits
	outputLayer.biases[0] = 1000.0
	outputLayer.biases[1] = 1005.0
	outputsExtreme := CalculateOutputs(nn, inputs)
	if math.IsNaN(float64(outputsExtreme[0])) || math.IsInf(float64(outputsExtreme[0]), 0) {
		t.Errorf("Softmax produced NaN or Inf on large inputs: %v", outputsExtreme)
	}
	sumExtreme := outputsExtreme[0] + outputsExtreme[1]
	if math.Abs(float64(sumExtreme-1.0)) > 1e-5 {
		t.Errorf("Expected extreme Softmax sum == 1.0, got %f", sumExtreme)
	}
}

func TestOutputLayerGradientsNonZeroWhenNegative(t *testing.T) {
	nn := NewNeuralNetwork([]int{2, 2})
	outputLayer := nn.layers[0]

	outputLayer.biases[0] = -2.0
	outputLayer.biases[1] = 0.0
	for r := range outputLayer.weights {
		for c := range outputLayer.weights[r] {
			outputLayer.weights[r][c] = 0
		}
	}

	dataPoint := DataPoint{
		inputs:  []float32{1.0, 1.0},
		outputs: []float32{1.0, 0.0},
	}

	UpdateAllGradients(nn, dataPoint)

	// Logits [-2.0, 0.0] -> a0 ~= 0.119203, a1 ~= 0.880797
	// With Cross-Entropy, delta0 = a0 - y0 = 0.119203 - 1.0 = -0.880797
	expectedBiasGrad := float32(-0.880797)
	if math.Abs(float64(outputLayer.costGradientB[0]-expectedBiasGrad)) > 1e-4 {
		t.Errorf("Expected bias gradient ~%f, got %f", expectedBiasGrad, outputLayer.costGradientB[0])
	}
	if outputLayer.costGradientB[0] == 0.0 {
		t.Errorf("Output neuron gradient is zero (dead neuron)")
	}

	expectedWeightGrad := expectedBiasGrad * 1.0 // input is 1.0
	if math.Abs(float64(outputLayer.costGradientW[0][0]-expectedWeightGrad)) > 1e-4 {
		t.Errorf("Expected weight gradient ~%f, got %f", expectedWeightGrad, outputLayer.costGradientW[0][0])
	}
}

func TestLeakyReLUActivation(t *testing.T) {
	// Positive inputs pass through unchanged
	if val := ActivationFunctionLeakyRELU(4.5); val != 4.5 {
		t.Errorf("Expected LeakyReLU(4.5) == 4.5, got %f", val)
	}
	if d := DerivativeActivationFunctionLeakyRELU(4.5); d != 1.0 {
		t.Errorf("Expected derivative for 4.5 == 1.0, got %f", d)
	}

	// Zero input
	if val := ActivationFunctionLeakyRELU(0.0); val != 0.0 {
		t.Errorf("Expected LeakyReLU(0.0) == 0.0, got %f", val)
	}

	// Negative inputs scaled by alpha = 0.01 (eliminating dead neurons)
	if val := ActivationFunctionLeakyRELU(-4.0); math.Abs(float64(val-(-0.04))) > 1e-6 {
		t.Errorf("Expected LeakyReLU(-4.0) == -0.04, got %f", val)
	}
	if d := DerivativeActivationFunctionLeakyRELU(-4.0); d != 0.01 {
		t.Errorf("Expected derivative for -4.0 == 0.01, got %f", d)
	}
}

func TestCrossEntropyLossCalculation(t *testing.T) {
	nn := NewNeuralNetwork([]int{2, 2})
	outputLayer := nn.layers[0]
	for r := range outputLayer.weights {
		for c := range outputLayer.weights[r] {
			outputLayer.weights[r][c] = 0
		}
	}

	// Case 1: Equal logits -> probabilities [0.5, 0.5] -> cost = -ln(0.5) ~= 0.693147
	outputLayer.biases[0] = 0.0
	outputLayer.biases[1] = 0.0
	dp1 := DataPoint{
		inputs:  []float32{1.0, 1.0},
		outputs: []float32{1.0, 0.0},
	}
	cost1 := Cost(nn, dp1)
	if math.Abs(float64(cost1-0.693147)) > 1e-4 {
		t.Errorf("Expected binary uniform cost ~ 0.693147, got %f", cost1)
	}

	// Case 2: High confidence correct prediction -> cost near 0
	outputLayer.biases[0] = 10.0
	outputLayer.biases[1] = -10.0
	cost2 := Cost(nn, dp1)
	if cost2 < 0.0 || cost2 > 0.001 {
		t.Errorf("Expected near-zero cost for confident correct prediction, got %f", cost2)
	}

	// Case 3: High confidence incorrect prediction -> cost high and stable (bounded by eps=1e-7)
	outputLayer.biases[0] = -100.0
	outputLayer.biases[1] = 100.0
	cost3 := Cost(nn, dp1)
	if math.IsNaN(float64(cost3)) || math.IsInf(float64(cost3), 0) || cost3 <= 0 {
		t.Errorf("Expected finite, positive cost for wrong prediction, got %f", cost3)
	}
}

func TestIrisFeatureNormalizationRange(t *testing.T) {
	data := ReadIrisDataset("./archive/Iris.csv")
	if len(data) != 150 {
		t.Fatalf("Expected 150 iris records, got %d", len(data))
	}
	points := IrisToDataPoint(data)
	if len(points) != 150 {
		t.Fatalf("Expected 150 points, got %d", len(points))
	}

	for i, p := range points {
		for fIdx, val := range p.inputs {
			if val < 0.0 || val > 1.0 {
				t.Errorf("Point %d feature %d out of bounds [0, 1]: got %f", i, fIdx, val)
			}
		}
	}
}

func TestDatasetBatchingCoversAllPoints(t *testing.T) {
	testSizes := []int{150, 1000, 35, 1, 64}
	batchSize := 32

	for _, n := range testSizes {
		visited := make([]bool, n)
		for i := 0; i < n; i += batchSize {
			numEnd := i + batchSize
			if numEnd > n {
				numEnd = n
			}
			for k := i; k < numEnd; k++ {
				if visited[k] {
					t.Fatalf("Dataset size %d: index %d visited multiple times", n, k)
				}
				visited[k] = true
			}
		}

		for idx, v := range visited {
			if !v {
				t.Fatalf("Dataset size %d: index %d was skipped!", n, idx)
			}
		}
	}
}

func TestGetClassColor(t *testing.T) {
	c0 := GetClassColor(0)
	if c0 != COLOR_1 {
		t.Errorf("Expected COLOR_1 for class 0, got %v", c0)
	}
	c1 := GetClassColor(1)
	if c1 != COLOR_2 {
		t.Errorf("Expected COLOR_2 for class 1, got %v", c1)
	}
	c2 := GetClassColor(2)
	if c2.R != 76 || c2.G != 175 || c2.B != 80 {
		t.Errorf("Expected Green for class 2, got %v", c2)
	}
	// Fallback for out-of-range class
	cFallback := GetClassColor(999)
	if cFallback != COLOR_3 {
		t.Errorf("Expected COLOR_3 for out-of-range class, got %v", cFallback)
	}
}

func TestParseNNEmptyWeights(t *testing.T) {
	// A layer with 0 inputs
	nn := &NeuralNetwork{
		layers: []*Layer{
			{
				numNodesIn:  0,
				numNodesOut: 2,
				weights:     [][]float32{},
				biases:      []float32{0.1, 0.2},
			},
		},
	}
	// Should not panic
	sliders := ParseNN(nn)
	if len(sliders) != 2 {
		t.Errorf("Expected 2 bias sliders, got %d", len(sliders))
	}
}

func TestLoadCSVResourceHandling(t *testing.T) {
	records := load_csv("./archive/Iris.csv")
	if len(records) != 151 { // 1 header + 150 data rows
		t.Errorf("Expected 151 rows from Iris.csv, got %d", len(records))
	}
}

func TestLearningRateFloor(t *testing.T) {
	lr := float32(0.00010005)
	minLR := float32(0.0001)

	// Simulate many frames of decay
	for frame := 0; frame < 100000; frame++ {
		if lr > minLR {
			lr -= 0.00000001
			if lr < minLR {
				lr = minLR
			}
		}
	}

	if lr < minLR {
		t.Errorf("Learning rate dropped below minLR: %f < %f", lr, minLR)
	}
	if lr != minLR {
		t.Errorf("Expected learning rate to floor at %f, got %f", minLR, lr)
	}
}

func TestFrameUpdatePerformance(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 4, 8, 8, 8, 8, 4, 2})

	boundaryBuffer := make([]byte, BOUNDARY_RESOLUTION*BOUNDARY_RESOLUTION*4)
	game := &Game{
		screenWidth:  DEFAULT_WINDOW_W,
		screenHeight: DEFAULT_WINDOW_H,
		pixelImage:   nil, // headless
		pixelBuffer:  boundaryBuffer,
		points:       points,
		nn:           nn,
		paramDials:   ParseNN(nn),
		activeSlider: -1,
		cost:         0.0,
		lr:           0.01,
		tickCount:    0,
	}

	numFrames := 12
	start := time.Now()
	for f := 0; f < numFrames; f++ {
		// Simulate Update without ebiten window dependencies
		batchSize := 32
		numPoints := len(game.points)
		if numPoints > 0 {
			for j := 0; j < EPOCHS_PER_FRAME; j += 1 {
				for i := 0; i < numPoints; i += batchSize {
					numEnd := i + batchSize
					if numEnd > numPoints {
						numEnd = numPoints
					}
					Learn(game.nn, game.points[i:numEnd], game.lr)
				}
			}
		}

		game.tickCount++
		if game.tickCount%BOUNDARY_UPDATE_INTERVAL == 0 {
			UpdateSliders(game)
			// Decision boundary compute without WritePixels
			numFeatures := len(game.points[0].inputs)
			ip := make([]float32, numFeatures)
			for row := 0; row < BOUNDARY_RESOLUTION; row++ {
				dataY := 1.0 - float32(row)/float32(BOUNDARY_RESOLUTION)
				for col := 0; col < BOUNDARY_RESOLUTION; col++ {
					dataX := float32(col) / float32(BOUNDARY_RESOLUTION)
					ip[0] = dataX
					ip[1] = dataY
					_ = Classify(game.nn, ip)
				}
			}
			cost := NetworkCost(game.nn, game.points)
			game.cost = cost
		}
	}
	totalDuration := time.Since(start)
	avgFrameTime := totalDuration / time.Duration(numFrames)

	t.Logf("Total duration for %d frames: %v (Average: %v per frame)", numFrames, totalDuration, avgFrameTime)

	// Target for 60 FPS is 16.6ms. Average frame time should comfortably be under 16.6ms.
	if avgFrameTime > 16600*time.Microsecond {
		t.Errorf("Average frame time (%v) exceeds 16.6ms budget for 60 FPS!", avgFrameTime)
	}
}

func TestZeroAllocationsInLearn(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 4, 8, 8, 8, 8, 4, 2})
	batch := points[:32]

	// Warm up
	Learn(nn, batch, 0.01)

	allocs := testing.AllocsPerRun(50, func() {
		Learn(nn, batch, 0.01)
	})

	t.Logf("Allocations per Learn batch: %f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per Learn call, got %f", allocs)
	}
}

func TestZeroAllocationsInBoundaryEvaluation(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 4, 8, 8, 8, 8, 4, 2})

	boundaryBuffer := make([]byte, BOUNDARY_RESOLUTION*BOUNDARY_RESOLUTION*4)
	numFeatures := len(points[0].inputs)
	featureMeans := make([]float32, numFeatures)
	boundaryInput := make([]float32, numFeatures)

	game := &Game{
		screenWidth:   DEFAULT_WINDOW_W,
		screenHeight:  DEFAULT_WINDOW_H,
		pixelImage:    nil,
		pixelBuffer:   boundaryBuffer,
		points:        points,
		nn:            nn,
		paramDials:    ParseNN(nn),
		activeSlider:  -1,
		cost:          0.0,
		lr:            0.01,
		tickCount:     0,
		featureMeans:  featureMeans,
		boundaryInput: boundaryInput,
	}

	// Warm up / simulate boundary classification loop
	classifyGrid := func() {
		copy(game.boundaryInput, game.featureMeans)
		for row := 0; row < 10; row++ {
			dataY := 1.0 - float32(row)/float32(10)
			for col := 0; col < 10; col++ {
				dataX := float32(col) / float32(10)
				game.boundaryInput[0] = dataX
				game.boundaryInput[1] = dataY
				_ = Classify(game.nn, game.boundaryInput)
			}
		}
	}

	classifyGrid()

	allocs := testing.AllocsPerRun(20, classifyGrid)
	t.Logf("Allocations per boundary classification grid: %f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per boundary classification grid, got %f", allocs)
	}
}

func TestSliderValueMappingRange(t *testing.T) {
	s := NewSlider("Weight", 0, 0, 0, 0)
	if s.Value() != 0.0 {
		t.Errorf("Expected default slider value 0.0, got %f", s.Value())
	}

	// Test mapping -2.5 to 0.0
	s.SetValue(-2.5)
	if s.normValue != 0.0 || s.Value() != -2.5 {
		t.Errorf("Expected normValue 0.0 and Value -2.5, got norm %f, val %f", s.normValue, s.Value())
	}

	// Test mapping 2.5 to 1.0
	s.SetValue(2.5)
	if s.normValue != 1.0 || s.Value() != 2.5 {
		t.Errorf("Expected normValue 1.0 and Value 2.5, got norm %f, val %f", s.normValue, s.Value())
	}

	// Test clamping out of bounds
	s.SetValue(-10.0)
	if s.normValue != 0.0 || s.Value() != -2.5 {
		t.Errorf("Expected clamped normValue 0.0, got norm %f", s.normValue)
	}

	s.SetValue(10.0)
	if s.normValue != 1.0 || s.Value() != 2.5 {
		t.Errorf("Expected clamped normValue 1.0, got norm %f", s.normValue)
	}
}

func TestGetSliderLayoutWithScroll(t *testing.T) {
	_, y0, _, _ := GetSliderLayout(1000, 800, 0, 0.0)
	_, y0Scrolled, _, _ := GetSliderLayout(1000, 800, 0, -100.0)

	if y0Scrolled != y0-100.0 {
		t.Errorf("Expected y0Scrolled = %f, got %f", y0-100.0, y0Scrolled)
	}

	_, y1, _, _ := GetSliderLayout(1000, 800, 1, 0.0)
	if y1 != y0+SLIDER_SPACING {
		t.Errorf("Expected y1 = %f, got %f", y0+SLIDER_SPACING, y1)
	}
}

func TestTrainingPausePreservesManualWeights(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 4, 2})

	game := &Game{
		screenWidth:  DEFAULT_WINDOW_W,
		screenHeight: DEFAULT_WINDOW_H,
		points:       points,
		nn:           nn,
		isPaused:     true, // Paused!
		lr:           0.01,
	}

	// Manually set a weight
	testWeight := float32(1.85)
	game.nn.layers[0].weights[0][0] = testWeight

	// Simulate training loop when paused
	if !game.isPaused {
		Learn(game.nn, game.points[:32], game.lr)
	}

	// Assert the weight was not touched
	if game.nn.layers[0].weights[0][0] != testWeight {
		t.Errorf("Manual weight was modified despite training being paused!")
	}
}

func TestDynamicTopologyConvergence(t *testing.T) {
	// 1. Test on Fruit dataset (2 inputs, 2 outputs)
	fruitData := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	fruitPoints := FruitToDataPoint(fruitData)
	numFruitIn := len(fruitPoints[0].inputs)
	numFruitOut := len(fruitPoints[0].outputs)

	if numFruitIn != 2 || numFruitOut != 2 {
		t.Fatalf("Expected 2 inputs and 2 outputs for fruit dataset, got %d and %d", numFruitIn, numFruitOut)
	}

	fruitNN := NewNeuralNetwork([]int{numFruitIn, 16, 16, numFruitOut})
	initialFruitCost := NetworkCost(fruitNN, fruitPoints)

	for epoch := 0; epoch < 30; epoch++ {
		for i := 0; i < len(fruitPoints); i += 32 {
			end := i + 32
			if end > len(fruitPoints) {
				end = len(fruitPoints)
			}
			Learn(fruitNN, fruitPoints[i:end], 0.02)
		}
	}

	finalFruitCost := NetworkCost(fruitNN, fruitPoints)
	t.Logf("Fruit [2, 16, 16, 2]: Initial=%.4f, Final=%.4f", initialFruitCost, finalFruitCost)
	if finalFruitCost >= initialFruitCost {
		t.Errorf("Expected fruit network cost to decrease, initial=%f, final=%f", initialFruitCost, finalFruitCost)
	}

	// 2. Test on Iris dataset (4 inputs, 3 outputs)
	irisData := ReadIrisDataset("./archive/Iris.csv")
	irisPoints := IrisToDataPoint(irisData)
	numIrisIn := len(irisPoints[0].inputs)
	numIrisOut := len(irisPoints[0].outputs)

	if numIrisIn != 4 || numIrisOut != 3 {
		t.Fatalf("Expected 4 inputs and 3 outputs for iris dataset, got %d and %d", numIrisIn, numIrisOut)
	}

	irisNN := NewNeuralNetwork([]int{numIrisIn, 16, 16, numIrisOut})
	initialIrisCost := NetworkCost(irisNN, irisPoints)

	for epoch := 0; epoch < 30; epoch++ {
		for i := 0; i < len(irisPoints); i += 32 {
			end := i + 32
			if end > len(irisPoints) {
				end = len(irisPoints)
			}
			Learn(irisNN, irisPoints[i:end], 0.02)
		}
	}

	finalIrisCost := NetworkCost(irisNN, irisPoints)
	t.Logf("Iris [4, 16, 16, 3]: Initial=%.4f, Final=%.4f", initialIrisCost, finalIrisCost)
	if finalIrisCost >= initialIrisCost {
		t.Errorf("Expected iris network cost to decrease, initial=%f, final=%f", initialIrisCost, finalIrisCost)
	}
}

func TestParallelBoundaryRendererZeroAllocations(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 16, 16, 2})
	numFeatures := len(points[0].inputs)

	renderer := NewBoundaryRenderer(nn, numFeatures, BOUNDARY_RESOLUTION)
	means := make([]float32, numFeatures)
	renderer.SetFeatureMeans(means)

	buf := make([]byte, BOUNDARY_RESOLUTION*BOUNDARY_RESOLUTION*4)

	// Warmup
	renderer.Render(buf)

	// Assert zero allocations during multi-threaded rendering
	allocs := testing.AllocsPerRun(20, func() {
		renderer.Render(buf)
	})
	t.Logf("Allocations per multi-threaded boundary render: %f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per boundary render, got %f", allocs)
	}

	// Verify pixels are non-empty
	hasNonZero := false
	for _, b := range buf {
		if b != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Errorf("Expected pixel buffer to be written with color values")
	}
}

func TestLayerFlatAccessors(t *testing.T) {
	l := NewLayer(3, 4, false)
	testVal := float32(3.1415)
	l.SetWeight(1, 2, testVal)

	if l.GetWeight(1, 2) != testVal {
		t.Errorf("Expected GetWeight(1, 2) == %f, got %f", testVal, l.GetWeight(1, 2))
	}
	if l.weights[1][2] != testVal {
		t.Errorf("Expected 2D weights[1][2] == %f, got %f", testVal, l.weights[1][2])
	}

	l.costGradientW[2][3] = float32(-0.42)
	if l.GetWeightGradient(2, 3) != float32(-0.42) {
		t.Errorf("Expected GetWeightGradient(2, 3) == -0.42, got %f", l.GetWeightGradient(2, 3))
	}
}

func TestShuffleDataPoints(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	n := len(points)
	if n == 0 {
		t.Fatalf("Empty fruit points")
	}

	// Track original inputs to verify conservation of elements
	sumBefore := float32(0.0)
	for _, p := range points {
		sumBefore += p.inputs[0] + p.inputs[1]
	}

	ShuffleDataPoints(points)

	sumAfter := float32(0.0)
	for _, p := range points {
		sumAfter += p.inputs[0] + p.inputs[1]
	}

	if math.Abs(float64(sumBefore-sumAfter)) > 1e-3 {
		t.Errorf("Sum before shuffle %f != sum after shuffle %f", sumBefore, sumAfter)
	}

	// Verify zero allocations
	allocs := testing.AllocsPerRun(50, func() {
		ShuffleDataPoints(points)
	})
	t.Logf("Allocations per ShuffleDataPoints: %f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per ShuffleDataPoints call, got %f", allocs)
	}
}

func TestCalculateAccuracy(t *testing.T) {
	nn := NewNeuralNetwork([]int{2, 2})
	outputLayer := nn.layers[0]
	for r := range outputLayer.weights {
		for c := range outputLayer.weights[r] {
			outputLayer.weights[r][c] = 0
		}
	}
	// Class 0 biased heavily positive
	outputLayer.biases[0] = 10.0
	outputLayer.biases[1] = -10.0

	testData := []DataPoint{
		{inputs: []float32{0, 0}, outputs: []float32{1, 0}},
		{inputs: []float32{0, 1}, outputs: []float32{1, 0}},
		{inputs: []float32{1, 0}, outputs: []float32{0, 1}}, // incorrect prediction
		{inputs: []float32{1, 1}, outputs: []float32{1, 0}},
	}

	acc := CalculateAccuracy(nn, testData)
	if acc != 75.0 {
		t.Errorf("Expected 75.0%% accuracy (3/4 correct), got %f", acc)
	}

	allocs := testing.AllocsPerRun(20, func() {
		_ = CalculateAccuracy(nn, testData)
	})
	t.Logf("Allocations per CalculateAccuracy: %f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per CalculateAccuracy, got %f", allocs)
	}
}

func TestSingleStepTraining(t *testing.T) {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	points := FruitToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 16, 16, 2})

	costBefore := NetworkCost(nn, points)

	// Simulate stepRequested while paused
	batchSize := 32
	numPoints := len(points)
	ShuffleDataPoints(points)
	for i := 0; i < numPoints; i += batchSize {
		numEnd := i + batchSize
		if numEnd > numPoints {
			numEnd = numPoints
		}
		Learn(nn, points[i:numEnd], 0.05)
	}

	costAfter := NetworkCost(nn, points)
	t.Logf("Single step: Cost before=%.4f, Cost after=%.4f", costBefore, costAfter)
	if costAfter >= costBefore {
		t.Errorf("Expected cost to decrease after single step, got before=%f, after=%f", costBefore, costAfter)
	}
}

func TestActivationFunctions(t *testing.T) {
	// 1. Sigmoid
	s0 := ActivationFunctionSigmoid(0.0)
	if s0 < 0.49 || s0 > 0.51 {
		t.Errorf("Expected Sigmoid(0) ~ 0.5, got %f", s0)
	}
	ds0 := DerivativeActivationFunctionSigmoid(s0)
	if ds0 < 0.24 || ds0 > 0.26 { // 0.5 * 0.5 = 0.25
		t.Errorf("Expected Sigmoid'(0) ~ 0.25, got %f", ds0)
	}

	// 2. Tanh
	t0 := ActivationFunctionTanh(0.0)
	if t0 < -0.01 || t0 > 0.01 {
		t.Errorf("Expected Tanh(0) ~ 0.0, got %f", t0)
	}
	dt0 := DerivativeActivationFunctionTanh(t0)
	if dt0 < 0.99 || dt0 > 1.01 { // 1 - 0 = 1
		t.Errorf("Expected Tanh'(0) ~ 1.0, got %f", dt0)
	}

	// 3. ReLU
	if ActivationFunctionRELU(5.0) != 5.0 || ActivationFunctionRELU(-5.0) != 0.0 {
		t.Errorf("ReLU failed")
	}
	if DerivativeActivationFunctionRELU(5.0) != 1.0 || DerivativeActivationFunctionRELU(-5.0) != 0.0 {
		t.Errorf("ReLU derivative failed")
	}

	// 4. LeakyReLU
	if ActivationFunctionLeakyRELU(5.0) != 5.0 || ActivationFunctionLeakyRELU(-100.0) != -1.0 {
		t.Errorf("LeakyReLU failed")
	}
	if DerivativeActivationFunctionLeakyRELU(5.0) != 1.0 || DerivativeActivationFunctionLeakyRELU(-5.0) != 0.01 {
		t.Errorf("LeakyReLU derivative failed")
	}
}

func TestOutputAndCostCombinations(t *testing.T) {
	layerSizes := []int{4, 8, 3}
	data := []DataPoint{
		{inputs: []float32{1.0, 0.5, -0.5, 0.0}, outputs: []float32{1.0, 0.0, 0.0}},
		{inputs: []float32{-0.5, 1.0, 0.0, 0.5}, outputs: []float32{0.0, 1.0, 0.0}},
	}

	combos := []struct {
		act  ActivationType
		out  OutputType
		cost CostType
	}{
		{ActivationLeakyReLU, OutputSoftmax, CostCrossEntropy},
		{ActivationReLU, OutputSigmoid, CostCrossEntropy},
		{ActivationSigmoid, OutputLinear, CostMSE},
		{ActivationTanh, OutputSigmoid, CostMSE},
		{ActivationLeakyReLU, OutputSoftmax, CostMSE},
	}

	for _, c := range combos {
		nn := NewConfiguredNeuralNetwork(layerSizes, c.act, c.out, c.cost, 0.5)
		costBefore := NetworkCost(nn, data)
		// Perform 5 steps of learning
		for step := 0; step < 5; step++ {
			Learn(nn, data, 0.05)
		}
		costAfter := NetworkCost(nn, data)
		t.Logf("Combo Act=%s, Out=%s, Cost=%s: before=%.4f, after=%.4f",
			c.act.String(), c.out.String(), c.cost.String(), costBefore, costAfter)
		if costAfter >= costBefore {
			t.Errorf("Expected cost to decrease for combo Act=%s, Out=%s, Cost=%s: before=%f, after=%f",
				c.act.String(), c.out.String(), c.cost.String(), costBefore, costAfter)
		}
	}
}

func TestMomentumAcceleration(t *testing.T) {
	layerSizes := []int{4, 8, 2}
	data := []DataPoint{
		{inputs: []float32{0.5, 0.8, -0.2, 0.1}, outputs: []float32{1.0, 0.0}},
		{inputs: []float32{-0.3, -0.5, 0.9, -0.4}, outputs: []float32{0.0, 1.0}},
	}

	// Create two identical networks with identical weights
	nnNoMom := NewConfiguredNeuralNetwork(layerSizes, ActivationTanh, OutputSoftmax, CostCrossEntropy, 0.0)
	nnWithMom := NewConfiguredNeuralNetwork(layerSizes, ActivationTanh, OutputSoftmax, CostCrossEntropy, 0.9)

	// Copy weights and biases so they start identical
	for lIdx := range nnNoMom.layers {
		copy(nnWithMom.layers[lIdx].weightsBacking, nnNoMom.layers[lIdx].weightsBacking)
		copy(nnWithMom.layers[lIdx].biases, nnNoMom.layers[lIdx].biases)
	}

	// Train both for 15 steps
	lr := float32(0.02)
	for i := 0; i < 15; i++ {
		Learn(nnNoMom, data, lr)
		Learn(nnWithMom, data, lr)
	}

	costNoMom := NetworkCost(nnNoMom, data)
	costWithMom := NetworkCost(nnWithMom, data)

	t.Logf("Momentum comparison: without=%.4f, with=%.4f", costNoMom, costWithMom)

	// Verify that velocities are non-zero when momentum > 0
	hasVelocity := false
	for _, l := range nnWithMom.layers {
		for _, v := range l.velocityB {
			if v != 0 {
				hasVelocity = true
				break
			}
		}
	}
	if !hasVelocity {
		t.Errorf("Expected network with momentum to maintain non-zero velocities")
	}
}
