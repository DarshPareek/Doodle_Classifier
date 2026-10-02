package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
)

type MNISTImage struct {
	Pixels []float32
	Label  int
	Rows   int
	Cols   int
}

func NewMNISTImage(pixels []float32, label int, rows, cols int) *MNISTImage {
	return &MNISTImage{
		Pixels: pixels,
		Label:  label,
		Rows:   rows,
		Cols:   cols,
	}
}

func resolveIDXPath(path string) (string, error) {
	fi, err := os.Stat(path)
	if err == nil {
		if !fi.IsDir() {
			return path, nil
		}
		// If directory, check candidate file inside
		sub := filepath.Join(path, filepath.Base(path))
		if sfi, err := os.Stat(sub); err == nil && !sfi.IsDir() {
			return sub, nil
		}
		entries, err := os.ReadDir(path)
		if err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					return filepath.Join(path, e.Name()), nil
				}
			}
		}
		return "", fmt.Errorf("no file found in directory: %s", path)
	}

	// Try checking inside archive/mnist/ if path was given as a relative or base name
	altPath := filepath.Join("archive", "mnist", path)
	if fi, err := os.Stat(altPath); err == nil {
		if !fi.IsDir() {
			return altPath, nil
		}
		sub := filepath.Join(altPath, filepath.Base(altPath))
		if sfi, err := os.Stat(sub); err == nil && !sfi.IsDir() {
			return sub, nil
		}
	}

	return "", fmt.Errorf("file not found: %s", path)
}

func ReadMNISTImages(path string, limit ...int) ([][]float32, int, int, error) {
	resolvedPath, err := resolveIDXPath(path)
	if err != nil {
		return nil, 0, 0, err
	}
	f, err := os.Open(resolvedPath)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	var header [16]byte
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read images header: %w", err)
	}

	magic := binary.BigEndian.Uint32(header[0:4])
	if magic != 2051 {
		return nil, 0, 0, fmt.Errorf("invalid images magic number: %d (expected 2051)", magic)
	}

	numImages := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	if len(limit) > 0 && limit[0] > 0 && limit[0] < numImages {
		numImages = limit[0]
	}

	imageSize := rows * cols
	images := make([][]float32, numImages)
	rawBuf := make([]byte, imageSize)

	for i := 0; i < numImages; i++ {
		if _, err := io.ReadFull(reader, rawBuf); err != nil {
			return nil, 0, 0, fmt.Errorf("failed to read image %d: %w", i, err)
		}
		img := make([]float32, imageSize)
		for j := 0; j < imageSize; j++ {
			img[j] = float32(rawBuf[j])
		}
		images[i] = img
	}

	return images, rows, cols, nil
}

func ReadMNISTLabels(path string, limit ...int) ([]int, error) {
	resolvedPath, err := resolveIDXPath(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(resolvedPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	var header [8]byte
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return nil, fmt.Errorf("failed to read labels header: %w", err)
	}

	magic := binary.BigEndian.Uint32(header[0:4])
	if magic != 2049 {
		return nil, fmt.Errorf("invalid labels magic number: %d (expected 2049)", magic)
	}

	numLabels := int(binary.BigEndian.Uint32(header[4:8]))
	if len(limit) > 0 && limit[0] > 0 && limit[0] < numLabels {
		numLabels = limit[0]
	}

	rawBuf := make([]byte, numLabels)
	if _, err := io.ReadFull(reader, rawBuf); err != nil {
		return nil, fmt.Errorf("failed to read labels: %w", err)
	}

	labels := make([]int, numLabels)
	for i := 0; i < numLabels; i++ {
		labels[i] = int(rawBuf[i])
	}

	return labels, nil
}

func ReadMNISTDatasetSafe(imagesPath, labelsPath string, limit ...int) ([]MNISTImage, error) {
	images, rows, cols, err := ReadMNISTImages(imagesPath, limit...)
	if err != nil {
		return nil, err
	}
	labels, err := ReadMNISTLabels(labelsPath, limit...)
	if err != nil {
		return nil, err
	}

	count := len(images)
	if len(labels) < count {
		count = len(labels)
	}

	dataset := make([]MNISTImage, count)
	for i := 0; i < count; i++ {
		dataset[i] = *NewMNISTImage(images[i], labels[i], rows, cols)
	}
	return dataset, nil
}

func ReadMNISTDataset(imagesPath, labelsPath string, limit ...int) []MNISTImage {
	dataset, err := ReadMNISTDatasetSafe(imagesPath, labelsPath, limit...)
	if err != nil {
		log.Fatalf("Error while loading MNIST dataset (%s, %s): %v", imagesPath, labelsPath, err)
	}
	return dataset
}

func LoadMNISTTrain(baseDir string, limit ...int) []MNISTImage {
	imagesPath := filepath.Join(baseDir, "train-images.idx3-ubyte")
	labelsPath := filepath.Join(baseDir, "train-labels.idx1-ubyte")
	return ReadMNISTDataset(imagesPath, labelsPath, limit...)
}

func LoadMNISTTest(baseDir string, limit ...int) []MNISTImage {
	imagesPath := filepath.Join(baseDir, "t10k-images.idx3-ubyte")
	labelsPath := filepath.Join(baseDir, "t10k-labels.idx1-ubyte")
	return ReadMNISTDataset(imagesPath, labelsPath, limit...)
}

func MNISTToDataPoint(images []MNISTImage) []DataPoint {
	dataPoints := make([]DataPoint, len(images))
	for i := 0; i < len(images); i++ {
		ip := make([]float32, len(images[i].Pixels))
		for j, p := range images[i].Pixels {
			if p > 1.0 {
				ip[j] = p / 255.0
			} else {
				ip[j] = p
			}
		}
		op := make([]float32, 10)
		lbl := images[i].Label
		if lbl >= 0 && lbl < 10 {
			op[lbl] = 1.0
		}
		dataPoints[i] = DataPoint{
			inputs:  ip,
			outputs: op,
		}
	}
	return dataPoints
}
