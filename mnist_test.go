package main

import (
	"math"
	"os"
	"testing"
)

func TestNewMNISTImage(t *testing.T) {
	pixels := make([]float32, 784)
	img := NewMNISTImage(pixels, 7, 28, 28)
	if img == nil {
		t.Fatal("Expected NewMNISTImage to return non-nil")
	}
	if img.Label != 7 {
		t.Errorf("Expected label 7, got %d", img.Label)
	}
	if img.Rows != 28 || img.Cols != 28 {
		t.Errorf("Expected 28x28, got %dx%d", img.Rows, img.Cols)
	}
	if len(img.Pixels) != 784 {
		t.Errorf("Expected 784 pixels, got %d", len(img.Pixels))
	}
}

func TestReadMNISTImagesAndLabels(t *testing.T) {
	imagesPath := "archive/mnist/t10k-images.idx3-ubyte"
	labelsPath := "archive/mnist/t10k-labels.idx1-ubyte"

	// 1. Read images with limit
	limit := 100
	images, rows, cols, err := ReadMNISTImages(imagesPath, limit)
	if err != nil {
		t.Fatalf("Failed to read MNIST images: %v", err)
	}
	if len(images) != limit {
		t.Errorf("Expected %d images, got %d", limit, len(images))
	}
	if rows != 28 || cols != 28 {
		t.Errorf("Expected 28x28 images, got %dx%d", rows, cols)
	}
	for i := 0; i < len(images); i++ {
		if len(images[i]) != 784 {
			t.Fatalf("Image %d: expected 784 pixels, got %d", i, len(images[i]))
		}
		for j := 0; j < len(images[i]); j++ {
			if images[i][j] < 0 || images[i][j] > 255 {
				t.Fatalf("Image %d, pixel %d out of range [0, 255]: %f", i, j, images[i][j])
			}
		}
	}

	// 2. Read labels with limit
	labels, err := ReadMNISTLabels(labelsPath, limit)
	if err != nil {
		t.Fatalf("Failed to read MNIST labels: %v", err)
	}
	if len(labels) != limit {
		t.Errorf("Expected %d labels, got %d", limit, len(labels))
	}
	for i := 0; i < len(labels); i++ {
		if labels[i] < 0 || labels[i] > 9 {
			t.Errorf("Label %d out of range [0, 9]: %d", i, labels[i])
		}
	}
}

func TestReadMNISTDataset(t *testing.T) {
	imagesPath := "archive/mnist/t10k-images.idx3-ubyte"
	labelsPath := "archive/mnist/t10k-labels.idx1-ubyte"

	limit := 50
	dataset := ReadMNISTDataset(imagesPath, labelsPath, limit)
	if len(dataset) != limit {
		t.Fatalf("Expected %d dataset entries, got %d", limit, len(dataset))
	}

	for i, entry := range dataset {
		if entry.Rows != 28 || entry.Cols != 28 {
			t.Errorf("Entry %d: expected 28x28, got %dx%d", i, entry.Rows, entry.Cols)
		}
		if len(entry.Pixels) != 784 {
			t.Errorf("Entry %d: expected 784 pixels, got %d", i, len(entry.Pixels))
		}
		if entry.Label < 0 || entry.Label > 9 {
			t.Errorf("Entry %d: invalid label %d", i, entry.Label)
		}
	}
}

func TestLoadMNISTTrainAndTestHelpers(t *testing.T) {
	baseDir := "archive/mnist"

	trainData := LoadMNISTTrain(baseDir, 20)
	if len(trainData) != 20 {
		t.Errorf("Expected 20 train items, got %d", len(trainData))
	}

	testData := LoadMNISTTest(baseDir, 20)
	if len(testData) != 20 {
		t.Errorf("Expected 20 test items, got %d", len(testData))
	}
}

func TestMNISTToDataPoint(t *testing.T) {
	dummyPixels := make([]float32, 784)
	dummyPixels[0] = 0.0
	dummyPixels[1] = 127.5
	dummyPixels[2] = 255.0

	images := []MNISTImage{
		{
			Pixels: dummyPixels,
			Label:  3,
			Rows:   28,
			Cols:   28,
		},
		{
			Pixels: dummyPixels,
			Label:  0,
			Rows:   28,
			Cols:   28,
		},
	}

	points := MNISTToDataPoint(images)
	if len(points) != 2 {
		t.Fatalf("Expected 2 data points, got %d", len(points))
	}

	// Verify normalization
	if points[0].inputs[0] != 0.0 {
		t.Errorf("Expected input[0] == 0.0, got %f", points[0].inputs[0])
	}
	if points[0].inputs[1] < 0.49 || points[0].inputs[1] > 0.51 {
		t.Errorf("Expected input[1] ~ 0.5, got %f", points[0].inputs[1])
	}
	if points[0].inputs[2] != 1.0 {
		t.Errorf("Expected input[2] == 1.0, got %f", points[0].inputs[2])
	}

	// Verify one-hot encoding for label 3
	if len(points[0].outputs) != 10 {
		t.Fatalf("Expected 10 outputs, got %d", len(points[0].outputs))
	}
	if points[0].outputs[3] != 1.0 {
		t.Errorf("Expected output[3] == 1.0 for label 3, got %f", points[0].outputs[3])
	}
	for k := 0; k < 10; k++ {
		if k != 3 && points[0].outputs[k] != 0.0 {
			t.Errorf("Expected output[%d] == 0.0, got %f", k, points[0].outputs[k])
		}
	}

	// Verify one-hot encoding for label 0
	if points[1].outputs[0] != 1.0 {
		t.Errorf("Expected output[0] == 1.0 for label 0, got %f", points[1].outputs[0])
	}
}

func TestReadMNISTErrors(t *testing.T) {
	// Non-existent path
	_, _, _, err := ReadMNISTImages("nonexistent_images_path")
	if err == nil {
		t.Errorf("Expected error for non-existent image path")
	}

	_, err = ReadMNISTLabels("nonexistent_labels_path")
	if err == nil {
		t.Errorf("Expected error for non-existent label path")
	}

	// Invalid magic number check
	tmpFile, err := os.CreateTemp("", "bad_magic_*.idx")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	badHeader := []byte{0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 28, 0, 0, 0, 28}
	tmpFile.Write(badHeader)
	tmpFile.Close()

	_, _, _, err = ReadMNISTImages(tmpFile.Name())
	if err == nil {
		t.Errorf("Expected error for invalid magic number")
	}
}

func TestMNISTGameInitializationAndLearning(t *testing.T) {
	data := ReadMNISTDataset("./archive/mnist/train-images.idx3-ubyte", "./archive/mnist/train-labels.idx1-ubyte", 50)
	points := MNISTToDataPoint(data)
	if len(points) != 50 {
		t.Fatalf("Expected 50 data points, got %d", len(points))
	}

	numInputs := len(points[0].inputs)
	numOutputs := len(points[0].outputs)
	if numInputs != 784 || numOutputs != 10 {
		t.Fatalf("Expected 784 inputs and 10 outputs, got %d and %d", numInputs, numOutputs)
	}

	layerSizes := []int{numInputs, 16, 16, numOutputs}
	nn := NewNeuralNetwork(layerSizes)

	trainPts, testPts := SplitDataset(points, 80)
	if len(trainPts) != 40 || len(testPts) != 10 {
		t.Fatalf("Expected 40 train, 10 test, got %d and %d", len(trainPts), len(testPts))
	}

	trainAcc := CalculateAccuracy(nn, trainPts)
	testAcc := CalculateAccuracy(nn, testPts)
	if trainAcc < 0 || trainAcc > 100 || testAcc < 0 || testAcc > 100 {
		t.Errorf("Accuracies out of range: train=%f, test=%f", trainAcc, testAcc)
	}

	costBefore := NetworkCost(nn, trainPts)
	// Run 1 learning step
	Learn(nn, trainPts, 0.05)
	costAfter := NetworkCost(nn, trainPts)
	t.Logf("MNIST step cost: before=%f, after=%f", costBefore, costAfter)
}

func TestMNISTTrainingCollapse(t *testing.T) {
	data := ReadMNISTDataset("./archive/mnist/train-images.idx3-ubyte", "./archive/mnist/train-labels.idx1-ubyte", 200)
	points := MNISTToDataPoint(data)
	trainPts, testPts := SplitDataset(points, 80)

	layerSizes := []int{784, 16, 32, 16, 10}
	nn := NewNeuralNetwork(layerSizes)
	lr := float32(1.0)
	batchSize := 32

	for epoch := 1; epoch <= 60; epoch++ {
		ShuffleDataPoints(trainPts)
		for i := 0; i < len(trainPts); i += batchSize {
			end := i + batchSize
			if end > len(trainPts) {
				end = len(trainPts)
			}
			Learn(nn, trainPts[i:end], lr)
		}
		for _, l := range nn.layers {
			for _, row := range l.weights {
				for _, w := range row {
					if w != w || math.IsInf(float64(w), 0) {
						t.Fatalf("Epoch %d: weight became NaN/Inf", epoch)
					}
				}
			}
		}
	}
	trainAcc := CalculateAccuracy(nn, trainPts)
	testAcc := CalculateAccuracy(nn, testPts)
	t.Logf("Post-training with high lr: TrainAcc=%.1f%%, TestAcc=%.1f%% (no NaN/Inf collapse)", trainAcc, testAcc)
}

func TestMNISTStableLearning(t *testing.T) {
	data := ReadMNISTDataset("./archive/mnist/train-images.idx3-ubyte", "./archive/mnist/train-labels.idx1-ubyte", 200)
	points := MNISTToDataPoint(data)
	trainPts, testPts := SplitDataset(points, 80)

	layerSizes := []int{784, 32, 10}
	nn := NewNeuralNetwork(layerSizes)
	lr := float32(0.05)
	batchSize := 16

	costInitial := NetworkCost(nn, trainPts)
	for epoch := 1; epoch <= 20; epoch++ {
		ShuffleDataPoints(trainPts)
		for i := 0; i < len(trainPts); i += batchSize {
			end := i + batchSize
			if end > len(trainPts) {
				end = len(trainPts)
			}
			Learn(nn, trainPts[i:end], lr)
		}
	}
	costFinal := NetworkCost(nn, trainPts)
	trainAcc := CalculateAccuracy(nn, trainPts)
	testAcc := CalculateAccuracy(nn, testPts)

	t.Logf("Stable learning: costInitial=%.4f, costFinal=%.4f, TrainAcc=%.1f%%, TestAcc=%.1f%%",
		costInitial, costFinal, trainAcc, testAcc)

	if costFinal >= costInitial {
		t.Errorf("Expected cost to decrease, was %.4f -> %.4f", costInitial, costFinal)
	}
	if trainAcc < 20.0 {
		t.Errorf("Expected train accuracy to improve above random guess, got %.1f%%", trainAcc)
	}
}


