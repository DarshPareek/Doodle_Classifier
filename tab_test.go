package main

import (
	"strings"
	"testing"
)

func TestTabConstants(t *testing.T) {
	if TAB_BAR_HEIGHT <= 0 {
		t.Errorf("Expected TAB_BAR_HEIGHT > 0, got %f", TAB_BAR_HEIGHT)
	}
	if TAB_WIDTH <= 0 {
		t.Errorf("Expected TAB_WIDTH > 0, got %f", TAB_WIDTH)
	}
}

func TestTabTypeEnum(t *testing.T) {
	if TabDecisionBoundary != 0 {
		t.Errorf("Expected TabDecisionBoundary == 0, got %d", TabDecisionBoundary)
	}
	if TabNewFeature != 1 {
		t.Errorf("Expected TabNewFeature == 1, got %d", TabNewFeature)
	}
}

func TestSliderLayoutOffsetWithTabs(t *testing.T) {
	trackX, trackY, trackW, trackH := GetSliderLayout(1000, 800, 0, 0)
	expectedMarginTop := TAB_BAR_HEIGHT + TOP_BAR_HEIGHT + 25
	if trackY != expectedMarginTop {
		t.Errorf("Expected trackY == %f, got %f", expectedMarginTop, trackY)
	}
	if trackW != SLIDER_PANEL_WIDTH-30 {
		t.Errorf("Expected trackW == %f, got %f", SLIDER_PANEL_WIDTH-30, trackW)
	}
	if trackH != 14 {
		t.Errorf("Expected trackH == 14, got %f", trackH)
	}
	if trackX != 1000-SLIDER_PANEL_WIDTH+15 {
		t.Errorf("Expected trackX == %f, got %f", 1000-SLIDER_PANEL_WIDTH+15, trackX)
	}
}

func TestGameTabSwitching(t *testing.T) {
	game := &Game{
		screenWidth:  DEFAULT_WINDOW_W,
		screenHeight: DEFAULT_WINDOW_H,
		currentTab:   TabDecisionBoundary,
	}

	if game.currentTab != TabDecisionBoundary {
		t.Errorf("Expected initial tab to be TabDecisionBoundary, got %d", game.currentTab)
	}

	game.currentTab = TabNewFeature
	if game.currentTab != TabNewFeature {
		t.Errorf("Expected tab to switch to TabNewFeature, got %d", game.currentTab)
	}
}

func TestToggleLearning(t *testing.T) {
	game := &Game{
		isLearning: false,
		isKilled:   true,
	}

	// Turn learning on
	game.ToggleLearning()
	if !game.isLearning {
		t.Errorf("Expected isLearning == true after toggle")
	}
	if game.isKilled {
		t.Errorf("Expected isKilled == false when learning is started")
	}

	// Turn learning off
	game.ToggleLearning()
	if game.isLearning {
		t.Errorf("Expected isLearning == false after second toggle")
	}
}

func TestRecordHistory(t *testing.T) {
	game := &Game{
		totalEpochs:   0,
		trainAccuracy: 50.0,
		testAccuracy:  48.0,
		history:       []HistoryPoint{{Epoch: 0, TrainAccuracy: 50.0, TestAccuracy: 48.0}},
	}

	// Update current epoch accuracy
	game.trainAccuracy = 55.0
	game.testAccuracy = 52.0
	game.RecordHistory()
	if len(game.history) != 1 {
		t.Errorf("Expected len(history) == 1 for same epoch, got %d", len(game.history))
	}
	if game.history[0].TrainAccuracy != 55.0 || game.history[0].TestAccuracy != 52.0 {
		t.Errorf("Expected updated Train 55.0, Test 52.0, got %+v", game.history[0])
	}

	// Record new epoch
	game.totalEpochs = 10
	game.trainAccuracy = 75.0
	game.testAccuracy = 70.0
	game.RecordHistory()
	if len(game.history) != 2 {
		t.Errorf("Expected len(history) == 2, got %d", len(game.history))
	}
	if game.history[1].Epoch != 10 || game.history[1].TrainAccuracy != 75.0 || game.history[1].TestAccuracy != 70.0 {
		t.Errorf("Expected epoch 10 with Train 75.0, Test 70.0, got %+v", game.history[1])
	}
}

func TestKillNetwork(t *testing.T) {
	layerSizes := []int{2, 4, 2}
	nn := NewNeuralNetwork(layerSizes)
	points := []DataPoint{
		{inputs: []float32{0.5, 0.5}, outputs: []float32{1.0, 0.0}},
	}
	game := &Game{
		layerSizes:   layerSizes,
		nn:           nn,
		points:       points,
		featureMeans: []float32{0.5, 0.5},
		isLearning:   true,
		totalEpochs:  100,
		history:      []HistoryPoint{{Epoch: 0, TrainAccuracy: 50.0, TestAccuracy: 50.0}, {Epoch: 100, TrainAccuracy: 90.0, TestAccuracy: 85.0}},
	}

	game.KillNetwork()

	if game.isLearning {
		t.Errorf("Expected isLearning == false after KillNetwork")
	}
	if !game.isKilled {
		t.Errorf("Expected isKilled == true after KillNetwork")
	}
	if game.totalEpochs != 0 {
		t.Errorf("Expected totalEpochs == 0 after KillNetwork, got %d", game.totalEpochs)
	}
	if len(game.history) != 1 || game.history[0].Epoch != 0 {
		t.Errorf("Expected history to be reset to epoch 0, got %+v", game.history)
	}
	if game.nn == nil {
		t.Errorf("Expected new neural network after kill, got nil")
	}
}

func TestParseLayerSizes(t *testing.T) {
	// Test valid formats: input size 2, hidden 16, 32, 16, output 2
	res, err := ParseLayerSizes("2, 16, 32, 16", 2)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expected := []int{2, 16, 32, 16, 2}
	if len(res) != len(expected) {
		t.Fatalf("Expected len %d, got %d", len(expected), len(res))
	}
	for i, v := range expected {
		if res[i] != v {
			t.Errorf("Index %d: expected %d, got %d", i, v, res[i])
		}
	}

	// Space separated with explicit output layer: input 4, hidden 8, 16, 8, output 3
	res2, err := ParseLayerSizes("4 8 16 8 3", 3)
	if err != nil || len(res2) != 5 || res2[0] != 4 || res2[1] != 8 || res2[2] != 16 || res2[3] != 8 || res2[4] != 3 {
		t.Errorf("Failed space-separated parse: %v, %v", res2, err)
	}

	// Single input layer size (no hidden layers): input 2, output 1
	res3, err := ParseLayerSizes("2", 1)
	if err != nil || len(res3) != 2 || res3[0] != 2 || res3[1] != 1 {
		t.Errorf("Failed single layer parse: %v, %v", res3, err)
	}

	// Truncated input size (1) with hidden layer 32 and output 2
	res4, err := ParseLayerSizes("1, 32", 2)
	if err != nil || len(res4) != 3 || res4[0] != 1 || res4[1] != 32 || res4[2] != 2 {
		t.Errorf("Failed truncated input parse: %v, %v", res4, err)
	}

	// Padded input size (5) with hidden layer 32 and output 2
	res5, err := ParseLayerSizes("5, 32", 2)
	if err != nil || len(res5) != 3 || res5[0] != 5 || res5[1] != 32 || res5[2] != 2 {
		t.Errorf("Failed padded input parse: %v, %v", res5, err)
	}

	// MNIST layer size (784 features, 10 classes)
	resMNIST, err := ParseLayerSizes("784, 16, 32, 16", 10)
	if err != nil || len(resMNIST) != 5 || resMNIST[0] != 784 || resMNIST[4] != 10 {
		t.Errorf("Failed MNIST 784 layer parse: %v, %v", resMNIST, err)
	}

	// Invalid inputs
	if _, err := ParseLayerSizes("", 2); err == nil {
		t.Errorf("Expected error for empty input")
	}
	if _, err := ParseLayerSizes("abc", 2); err == nil {
		t.Errorf("Expected error for non-numeric input")
	}
	if _, err := ParseLayerSizes("-5", 2); err == nil {
		t.Errorf("Expected error for negative size")
	}
	if _, err := ParseLayerSizes("9999", 2); err == nil {
		t.Errorf("Expected error for oversized layer")
	}
}

func TestAdjustDataPoints(t *testing.T) {
	orig := []DataPoint{
		{inputs: []float32{0.2, 0.4, 0.6, 0.8}, outputs: []float32{1.0, 0.0}},
	}

	// 1. Truncate: 4 -> 2
	trunc := AdjustDataPoints(orig, 2)
	if len(trunc[0].inputs) != 2 {
		t.Fatalf("Expected 2 inputs after truncation, got %d", len(trunc[0].inputs))
	}
	if trunc[0].inputs[0] != 0.2 || trunc[0].inputs[1] != 0.4 {
		t.Errorf("Unexpected truncated values: %+v", trunc[0].inputs)
	}

	// 2. Pad: 4 -> 6
	padded := AdjustDataPoints(orig, 6)
	if len(padded[0].inputs) != 6 {
		t.Fatalf("Expected 6 inputs after padding, got %d", len(padded[0].inputs))
	}
	if padded[0].inputs[0] != 0.2 || padded[0].inputs[1] != 0.4 || padded[0].inputs[2] != 0.6 || padded[0].inputs[3] != 0.8 {
		t.Errorf("Original values not preserved in padding: %+v", padded[0].inputs)
	}
	if padded[0].inputs[4] != 0.0 || padded[0].inputs[5] != 0.0 {
		t.Errorf("Padded values are not 0.0: %+v", padded[0].inputs)
	}

	// 3. Exact match: 4 -> 4
	exact := AdjustDataPoints(orig, 4)
	if len(exact[0].inputs) != 4 || exact[0].inputs[2] != 0.6 {
		t.Errorf("Unexpected exact match values: %+v", exact[0].inputs)
	}
}

func TestSplitDataset(t *testing.T) {
	pts := make([]DataPoint, 100)
	for i := range pts {
		pts[i] = DataPoint{inputs: []float32{float32(i)}}
	}

	train, test := SplitDataset(pts, 80)
	if len(train) != 80 || len(test) != 20 {
		t.Errorf("Expected 80 train / 20 test, got %d / %d", len(train), len(test))
	}

	train50, test50 := SplitDataset(pts, 50)
	if len(train50) != 50 || len(test50) != 50 {
		t.Errorf("Expected 50 train / 50 test, got %d / %d", len(train50), len(test50))
	}

	// Clamping check: 5% should clamp to 10%
	trainLow, testLow := SplitDataset(pts, 5)
	if len(trainLow) != 10 || len(testLow) != 90 {
		t.Errorf("Expected clamped 10/90, got %d/%d", len(trainLow), len(testLow))
	}
}

func TestRunNewNetworkGuard(t *testing.T) {
	pts := []DataPoint{
		{inputs: []float32{0.5, 0.5}, outputs: []float32{1.0, 0.0}},
		{inputs: []float32{0.1, 0.9}, outputs: []float32{0.0, 1.0}},
	}
	game := &Game{
		allPoints:         pts,
		points:            pts,
		trainSplitPercent: 80,
		inputLayerSizes:   "2, 8, 8",
		inputLearnRate:    "0.01",
		inputBatchSize:    "2",
		isKilled:          false, // Network is NOT killed
		isLearning:        true,
	}

	// Should fail because isKilled is false
	err := game.RunNewNetwork()
	if err == nil {
		t.Errorf("Expected RunNewNetwork to fail when isKilled is false")
	}
	if game.configError == "" {
		t.Errorf("Expected configError to be set when network is not killed")
	}

	// Now kill network and retry
	game.isKilled = true
	err = game.RunNewNetwork()
	if err != nil {
		t.Fatalf("Expected RunNewNetwork to succeed when isKilled is true, got: %v", err)
	}
	if !game.isLearning {
		t.Errorf("Expected new network to be in learning state")
	}
	if game.isKilled {
		t.Errorf("Expected isKilled to be false after running new network")
	}
	if game.totalEpochs != 0 {
		t.Errorf("Expected totalEpochs to start at 0, got %d", game.totalEpochs)
	}
	if len(game.nn.layers) != 3 { // 2 input -> 8 -> 8 -> 2 output => 3 layers
		t.Errorf("Expected 3 layers for hidden sizes [8, 8], got %d", len(game.nn.layers))
	}
	if len(game.history) != 1 {
		t.Fatalf("Expected history length 1 after RunNewNetwork, got %d", len(game.history))
	}
	if game.history[0].TrainAccuracy != game.trainAccuracy || game.history[0].TestAccuracy != game.testAccuracy {
		t.Errorf("Expected history[0] to match game.trainAccuracy and testAccuracy, got %+v", game.history[0])
	}
}

func TestTrainAndTestAccuracyTracking(t *testing.T) {
	pts := make([]DataPoint, 20)
	for i := 0; i < 10; i++ {
		pts[i] = DataPoint{inputs: []float32{1.0, 0.0}, outputs: []float32{1.0, 0.0}}
	}
	for i := 10; i < 20; i++ {
		pts[i] = DataPoint{inputs: []float32{0.0, 1.0}, outputs: []float32{0.0, 1.0}}
	}

	game := &Game{
		allPoints:         pts,
		points:            pts,
		trainSplitPercent: 70, // 14 train, 6 test
		inputLayerSizes:   "2, 4",
		inputLearnRate:    "0.05",
		inputBatchSize:    "4",
		isKilled:          true,
	}

	err := game.RunNewNetwork()
	if err != nil {
		t.Fatalf("Unexpected error running new network: %v", err)
	}

	if len(game.trainPoints) != 14 {
		t.Errorf("Expected 14 train points, got %d", len(game.trainPoints))
	}
	if len(game.testPoints) != 6 {
		t.Errorf("Expected 6 test points, got %d", len(game.testPoints))
	}

	// Verify both accuracies are tracked
	if game.trainAccuracy < 0 || game.trainAccuracy > 100 {
		t.Errorf("Invalid trainAccuracy: %f", game.trainAccuracy)
	}
	if game.testAccuracy < 0 || game.testAccuracy > 100 {
		t.Errorf("Invalid testAccuracy: %f", game.testAccuracy)
	}

	// Verify history point has both train and test accuracy
	pt := game.history[0]
	if pt.TrainAccuracy != game.trainAccuracy || pt.TestAccuracy != game.testAccuracy {
		t.Errorf("History point does not match current accuracies: %+v", pt)
	}
}

func TestRunNewNetworkWithTruncationAndPadding(t *testing.T) {
	// Raw points have 3 input features and 2 outputs
	pts := []DataPoint{
		{inputs: []float32{0.2, 0.5, 0.8}, outputs: []float32{1.0, 0.0}},
		{inputs: []float32{0.8, 0.3, 0.1}, outputs: []float32{0.0, 1.0}},
	}

	// 1. Truncate: input layer size is 1 (< 3 features)
	gameTrunc := &Game{
		allPoints:         pts,
		points:            pts,
		trainSplitPercent: 50,
		inputLayerSizes:   "1, 8, 8",
		inputLearnRate:    "0.01",
		inputBatchSize:    "1",
		isKilled:          true,
	}

	if err := gameTrunc.RunNewNetwork(); err != nil {
		t.Fatalf("Failed to run network with truncated input size 1: %v", err)
	}
	if gameTrunc.nn.layers[0].numNodesIn != 1 {
		t.Errorf("Expected network input nodes 1, got %d", gameTrunc.nn.layers[0].numNodesIn)
	}
	if len(gameTrunc.trainPoints[0].inputs) != 1 {
		t.Errorf("Expected train points to have 1 input feature, got %d", len(gameTrunc.trainPoints[0].inputs))
	}
	if gameTrunc.trainPoints[0].inputs[0] != 0.2 {
		t.Errorf("Expected truncated feature 0.2, got %f", gameTrunc.trainPoints[0].inputs[0])
	}

	// 2. Pad: input layer size is 5 (> 3 features)
	gamePad := &Game{
		allPoints:         pts,
		points:            pts,
		trainSplitPercent: 50,
		inputLayerSizes:   "5, 8, 8",
		inputLearnRate:    "0.01",
		inputBatchSize:    "1",
		isKilled:          true,
	}

	if err := gamePad.RunNewNetwork(); err != nil {
		t.Fatalf("Failed to run network with padded input size 5: %v", err)
	}
	if gamePad.nn.layers[0].numNodesIn != 5 {
		t.Errorf("Expected network input nodes 5, got %d", gamePad.nn.layers[0].numNodesIn)
	}
	if len(gamePad.trainPoints[0].inputs) != 5 {
		t.Errorf("Expected train points to have 5 input features, got %d", len(gamePad.trainPoints[0].inputs))
	}
	if gamePad.trainPoints[0].inputs[3] != 0.0 || gamePad.trainPoints[0].inputs[4] != 0.0 {
		t.Errorf("Expected padded features to be 0.0, got %+v", gamePad.trainPoints[0].inputs)
	}
}

func TestDropdownWidget(t *testing.T) {
	d := NewDropdown("Test", []string{"A", "B", "C"}, 1)
	if d.SelectedValue() != "B" {
		t.Errorf("Expected selected value 'B', got '%s'", d.SelectedValue())
	}
	d.SelectIndex(2)
	if d.SelectedValue() != "C" {
		t.Errorf("Expected selected value 'C', got '%s'", d.SelectedValue())
	}
	// Out of bounds index should not panic or change
	d.SelectIndex(99)
	if d.SelectedValue() != "C" {
		t.Errorf("Expected selected value 'C', got '%s'", d.SelectedValue())
	}
}

func TestMomentumValidation(t *testing.T) {
	pts := []DataPoint{
		{inputs: []float32{1.0, 2.0}, outputs: []float32{1.0, 0.0}},
		{inputs: []float32{3.0, 4.0}, outputs: []float32{0.0, 1.0}},
	}
	game := &Game{
		allPoints:         pts,
		trainSplitPercent: 50,
		inputLayerSizes:   "2, 4, 2",
		inputLearnRate:    "0.01",
		inputBatchSize:    "1",
		isKilled:          true,
	}

	// Invalid momentum tests
	invalidValues := []string{"-0.1", "1.0", "1.5", "abc"}
	for _, val := range invalidValues {
		game.inputMomentum = val
		err := game.RunNewNetwork()
		if err == nil {
			t.Errorf("Expected error for invalid momentum '%s'", val)
		}
	}

	// Valid momentum test
	game.inputMomentum = "0.95"
	if err := game.RunNewNetwork(); err != nil {
		t.Errorf("Expected success for valid momentum '0.95', got: %v", err)
	}
	if game.nn.momentum != 0.95 {
		t.Errorf("Expected nn momentum 0.95, got %f", game.nn.momentum)
	}
}

func TestRunNewNetworkWithCustomHyperparameters(t *testing.T) {
	pts := []DataPoint{
		{inputs: []float32{1.0, 2.0}, outputs: []float32{1.0, 0.0}},
		{inputs: []float32{3.0, 4.0}, outputs: []float32{0.0, 1.0}},
	}
	game := &Game{
		allPoints:          pts,
		trainSplitPercent:  50,
		inputLayerSizes:    "2, 8, 2",
		inputLearnRate:     "0.02",
		inputBatchSize:     "1",
		inputMomentum:      "0.85",
		dropdownActivation: NewDropdown("Hidden Activation", []string{"LeakyReLU", "ReLU", "Sigmoid", "Tanh"}, 3), // Tanh
		dropdownOutput:     NewDropdown("Output Function", []string{"Softmax", "Sigmoid", "Linear"}, 1),            // Sigmoid
		dropdownCost:       NewDropdown("Cost Function", []string{"Cross-Entropy", "MSE"}, 1),                      // MSE
		isKilled:           true,
	}

	if err := game.RunNewNetwork(); err != nil {
		t.Fatalf("Failed to run configured network: %v", err)
	}

	if game.nn.activationType != ActivationTanh {
		t.Errorf("Expected activation Tanh, got %v", game.nn.activationType)
	}
	if game.nn.outputType != OutputSigmoid {
		t.Errorf("Expected output Sigmoid, got %v", game.nn.outputType)
	}
	if game.nn.costType != CostMSE {
		t.Errorf("Expected cost MSE, got %v", game.nn.costType)
	}
	if game.nn.momentum != 0.85 {
		t.Errorf("Expected momentum 0.85, got %f", game.nn.momentum)
	}

	// Verify layers inherit settings
	for i, l := range game.nn.layers {
		if l.activationType != ActivationTanh {
			t.Errorf("Layer %d expected activation Tanh", i)
		}
		if l.outputType != OutputSigmoid {
			t.Errorf("Layer %d expected output Sigmoid", i)
		}
		if l.costType != CostMSE {
			t.Errorf("Layer %d expected cost MSE", i)
		}
		if l.momentum != 0.85 {
			t.Errorf("Layer %d expected momentum 0.85", i)
		}
	}
}

func TestTabDigitViewerEnum(t *testing.T) {
	if TabDigitViewer != 2 {
		t.Errorf("Expected TabDigitViewer == 2, got %d", TabDigitViewer)
	}
}

func TestViewerNavigation(t *testing.T) {
	images := []MNISTImage{
		{Pixels: make([]float32, 784), Label: 0, Rows: 28, Cols: 28},
		{Pixels: make([]float32, 784), Label: 1, Rows: 28, Cols: 28},
		{Pixels: make([]float32, 784), Label: 2, Rows: 28, Cols: 28},
	}
	game := &Game{
		mnistDataset: images,
		viewerIndex:  0,
	}

	if game.TotalViewerImages() != 3 {
		t.Errorf("Expected 3 images, got %d", game.TotalViewerImages())
	}

	game.NextViewerImage()
	if game.viewerIndex != 1 {
		t.Errorf("Expected index 1 after Next, got %d", game.viewerIndex)
	}

	game.NextViewerImage()
	if game.viewerIndex != 2 {
		t.Errorf("Expected index 2 after Next, got %d", game.viewerIndex)
	}

	// Wraparound on Next
	game.NextViewerImage()
	if game.viewerIndex != 0 {
		t.Errorf("Expected index 0 after Next wraparound, got %d", game.viewerIndex)
	}

	// Wraparound on Prev
	game.PrevViewerImage()
	if game.viewerIndex != 2 {
		t.Errorf("Expected index 2 after Prev wraparound, got %d", game.viewerIndex)
	}

	game.PrevViewerImage()
	if game.viewerIndex != 1 {
		t.Errorf("Expected index 1 after Prev, got %d", game.viewerIndex)
	}

	// Random stays within bounds
	for i := 0; i < 20; i++ {
		game.RandomViewerImage()
		if game.viewerIndex < 0 || game.viewerIndex >= 3 {
			t.Errorf("Random index out of bounds: %d", game.viewerIndex)
		}
	}
}

func TestViewerFallbackToAllPoints(t *testing.T) {
	pts := []DataPoint{
		{inputs: make([]float32, 784), outputs: []float32{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
	}
	game := &Game{
		allPoints: pts,
	}

	if game.TotalViewerImages() != 1 {
		t.Errorf("Expected 1 image from allPoints fallback, got %d", game.TotalViewerImages())
	}

	pixels, label, rows, cols := game.GetViewerSample(0)
	if len(pixels) != 784 || label != 1 || rows != 28 || cols != 28 {
		t.Errorf("Unexpected sample: len=%d, label=%d, rows=%d, cols=%d", len(pixels), label, rows, cols)
	}
}

func TestNormalizedProbabilities(t *testing.T) {
	// Case 1: Already normalized softmax outputs
	raw := []float32{0.7, 0.2, 0.1}
	probs := getNormalizedProbabilities(raw)
	if len(probs) != 3 {
		t.Fatalf("Expected len 3, got %d", len(probs))
	}
	if probs[0] < 0.69 || probs[0] > 0.71 {
		t.Errorf("Expected prob[0] ~ 0.7, got %f", probs[0])
	}

	// Case 2: Unnormalized logits (linear)
	logits := []float32{2.0, 1.0, 0.1}
	normProbs := getNormalizedProbabilities(logits)
	sum := normProbs[0] + normProbs[1] + normProbs[2]
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Expected sum ~ 1.0, got %f", sum)
	}
	if normProbs[0] <= normProbs[1] || normProbs[1] <= normProbs[2] {
		t.Errorf("Expected monotonic probabilities, got %v", normProbs)
	}
}

func TestFindNextMisclassified(t *testing.T) {
	nn := NewNeuralNetwork([]int{2, 2})
	// Force weights to predict class 0 always:
	nn.layers[0].weights[0][0] = 0
	nn.layers[0].weights[0][1] = 0
	nn.layers[0].weights[1][0] = 0
	nn.layers[0].weights[1][1] = 0
	nn.layers[0].biases[0] = 10.0
	nn.layers[0].biases[1] = -10.0

	// Dataset: sample 0 has label 0 (correct), sample 1 has label 1 (misclassified)
	images := []MNISTImage{
		{Pixels: []float32{0.5, 0.5}, Label: 0, Rows: 28, Cols: 28},
		{Pixels: []float32{0.5, 0.5}, Label: 1, Rows: 28, Cols: 28},
	}

	game := &Game{
		mnistDataset: images,
		layerSizes:   []int{2, 2},
		nn:           nn,
		viewerIndex:  0,
	}

	// Initial sample 0 is correctly classified as 0
	// FindNextMisclassified should jump to sample 1
	game.FindNextMisclassified()
	if game.viewerIndex != 1 {
		t.Errorf("Expected viewerIndex to be 1, got %d", game.viewerIndex)
	}
	if !strings.Contains(game.viewerStatusMsg, "Found misclassified #2") {
		t.Errorf("Expected status message about misclassified #2, got: %s", game.viewerStatusMsg)
	}

	// When all images are correctly classified
	imagesAllCorrect := []MNISTImage{
		{Pixels: []float32{0.5, 0.5}, Label: 0, Rows: 28, Cols: 28},
		{Pixels: []float32{0.5, 0.5}, Label: 0, Rows: 28, Cols: 28},
	}
	game2 := &Game{
		mnistDataset: imagesAllCorrect,
		layerSizes:   []int{2, 2},
		nn:           nn,
		viewerIndex:  0,
	}
	game2.FindNextMisclassified()
	if !strings.Contains(game2.viewerStatusMsg, "100%") {
		t.Errorf("Expected status message about 100%% accuracy, got: %s", game2.viewerStatusMsg)
	}
}

func TestTabCanvasDrawEnum(t *testing.T) {
	if TabCanvasDraw != 3 {
		t.Errorf("Expected TabCanvasDraw == 3, got %d", TabCanvasDraw)
	}
}

func TestCanvasClear(t *testing.T) {
	game := &Game{
		drawPixels: make([]float32, 784),
	}
	// Paint some test pixels
	game.drawPixels[100] = 0.8
	game.drawPixels[200] = 1.0
	if game.isCanvasEmpty() {
		t.Errorf("Expected isCanvasEmpty == false when pixels are set")
	}

	game.ClearCanvas()
	if !game.isCanvasEmpty() {
		t.Errorf("Expected isCanvasEmpty == true after ClearCanvas")
	}
	for i, p := range game.drawPixels {
		if p != 0 {
			t.Errorf("Pixel %d not zeroed: %f", i, p)
		}
	}
}

func TestCanvasStrokeInterpolation(t *testing.T) {
	game := &Game{
		drawPixels: make([]float32, 784),
	}
	// Stroke across the canvas
	game.PaintCanvasStroke(5, 5, 20, 20)

	if game.isCanvasEmpty() {
		t.Fatalf("Expected canvas not empty after stroke")
	}

	// Verify pixels along the stroke are non-zero and clamped
	nonZeroCount := 0
	for _, p := range game.drawPixels {
		if p > 0 {
			nonZeroCount++
			if p > 1.0 {
				t.Errorf("Pixel value exceeds 1.0: %f", p)
			}
		}
	}

	if nonZeroCount < 20 {
		t.Errorf("Expected at least 20 painted pixels for long stroke, got %d", nonZeroCount)
	}

	// Check center of stroke around (12, 12)
	idx := 12*28 + 12
	if game.drawPixels[idx] <= 0 {
		t.Errorf("Expected pixel at (12, 12) along the stroke to be painted, got %f", game.drawPixels[idx])
	}
}

func TestCanvasInferencePrediction(t *testing.T) {
	layerSizes := []int{784, 16, 10}
	nn := NewNeuralNetwork(layerSizes)
	game := &Game{
		nn:         nn,
		layerSizes: layerSizes,
		drawPixels: make([]float32, 784),
	}

	// Paint a circle/digit shape
	game.PaintCanvasStroke(10, 6, 18, 6)
	game.PaintCanvasStroke(18, 6, 18, 22)
	game.PaintCanvasStroke(18, 22, 10, 22)
	game.PaintCanvasStroke(10, 22, 10, 6)

	inputs := game.getViewerNNInputs(game.drawPixels)
	if len(inputs) != 784 {
		t.Fatalf("Expected 784 inputs, got %d", len(inputs))
	}

	rawOutputs := CalculateOutputs(game.nn, inputs)
	if len(rawOutputs) != 10 {
		t.Fatalf("Expected 10 outputs, got %d", len(rawOutputs))
	}

	probs := getNormalizedProbabilities(rawOutputs)
	if len(probs) != 10 {
		t.Fatalf("Expected 10 probabilities, got %d", len(probs))
	}

	sum := float32(0)
	for _, p := range probs {
		if p < 0 || p > 1.0 {
			t.Errorf("Probability out of range [0, 1]: %f", p)
		}
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Probabilities do not sum to 1: %f", sum)
	}

	predDigit := IndexOfMaxValue(rawOutputs)
	if predDigit < 0 || predDigit > 9 {
		t.Errorf("Predicted digit out of range [0, 9]: %d", predDigit)
	}
}

func TestGameCanvasTabSwitching(t *testing.T) {
	game := &Game{
		currentTab: TabDecisionBoundary,
	}

	game.currentTab = TabCanvasDraw
	if game.currentTab != TabCanvasDraw {
		t.Errorf("Expected tab to be TabCanvasDraw, got %d", game.currentTab)
	}
}


