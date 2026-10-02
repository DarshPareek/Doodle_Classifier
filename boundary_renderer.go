package main

import (
	"runtime"
	"sync"
)

type InferenceContext struct {
	activations   [][]float32
	boundaryInput []float32
}

type WorkerJob struct {
	startRow int
	endRow   int
}

type BoundaryRenderer struct {
	numWorkers   int
	contexts     []*InferenceContext
	jobs         []chan WorkerJob
	done         sync.WaitGroup
	resolution   int
	nn           *NeuralNetwork
	featureMeans []float32
	pixelBuffer  []byte
}

func NewBoundaryRenderer(nn *NeuralNetwork, numFeatures, resolution int) *BoundaryRenderer {
	nw := runtime.NumCPU()
	if nw < 1 {
		nw = 1
	}
	r := &BoundaryRenderer{
		numWorkers:   nw,
		contexts:     make([]*InferenceContext, nw),
		jobs:         make([]chan WorkerJob, nw),
		resolution:   resolution,
		nn:           nn,
		featureMeans: make([]float32, numFeatures),
	}

	for i := 0; i < nw; i++ {
		acts := make([][]float32, len(nn.layers))
		for lIdx, l := range nn.layers {
			acts[lIdx] = make([]float32, l.numNodesOut)
		}
		r.contexts[i] = &InferenceContext{
			activations:   acts,
			boundaryInput: make([]float32, numFeatures),
		}
		r.jobs[i] = make(chan WorkerJob)
	}

	bgR := float64(BG_COLOR.R)
	bgG := float64(BG_COLOR.G)
	bgB := float64(BG_COLOR.B)
	srcA := 100.0 / 255.0
	invA := 1.0 - srcA

	for i := 0; i < nw; i++ {
		go func(workerIdx int) {
			ctx := r.contexts[workerIdx]
			for job := range r.jobs[workerIdx] {
				for row := job.startRow; row < job.endRow; row++ {
					dataY := 1.0 - float32(row)/float32(r.resolution)
					for col := 0; col < r.resolution; col++ {
						dataX := float32(col) / float32(r.resolution)
						copy(ctx.boundaryInput, r.featureMeans)
						if len(ctx.boundaryInput) > 0 {
							ctx.boundaryInput[0] = dataX
						}
						if len(ctx.boundaryInput) > 1 {
							ctx.boundaryInput[1] = dataY
						}

						// Forward pass
						prev := ctx.boundaryInput
						for lIdx := range r.nn.layers {
							l := r.nn.layers[lIdx]
							out := ctx.activations[lIdx]
							for nodeOut := 0; nodeOut < l.numNodesOut; nodeOut++ {
								z := l.biases[nodeOut]
								for nodeIn := 0; nodeIn < l.numNodesIn; nodeIn++ {
									z += prev[nodeIn] * l.weights[nodeIn][nodeOut]
								}
								if !l.isOutput {
									if z > 0 {
										out[nodeOut] = z
									} else {
										out[nodeOut] = 0.01 * z
									}
								} else {
									out[nodeOut] = z
								}
							}
							prev = out
						}

						// Argmax of output logits/probabilities
						lastOut := ctx.activations[len(r.nn.layers)-1]
						bestIdx := 0
						for k := 1; k < len(lastOut); k++ {
							if lastOut[k] > lastOut[bestIdx] {
								bestIdx = k
							}
						}

						classColor := GetClassColor(bestIdx)
						outR := byte(float64(classColor.R)*srcA + bgR*invA)
						outG := byte(float64(classColor.G)*srcA + bgG*invA)
						outB := byte(float64(classColor.B)*srcA + bgB*invA)

						idx := (row*r.resolution + col) * 4
						r.pixelBuffer[idx] = outR
						r.pixelBuffer[idx+1] = outG
						r.pixelBuffer[idx+2] = outB
						r.pixelBuffer[idx+3] = 255
					}
				}
				r.done.Done()
			}
		}(i)
	}

	return r
}

func (r *BoundaryRenderer) SetFeatureMeans(means []float32) {
	if len(r.featureMeans) != len(means) {
		r.featureMeans = make([]float32, len(means))
		for _, ctx := range r.contexts {
			ctx.boundaryInput = make([]float32, len(means))
		}
	}
	copy(r.featureMeans, means)
}

func (r *BoundaryRenderer) Render(pixelBuffer []byte) {
	r.pixelBuffer = pixelBuffer
	rowsPerWorker := (r.resolution + r.numWorkers - 1) / r.numWorkers
	r.done.Add(r.numWorkers)
	for w := 0; w < r.numWorkers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if end > r.resolution {
			end = r.resolution
		}
		if start > r.resolution {
			start = r.resolution
		}
		r.jobs[w] <- WorkerJob{startRow: start, endRow: end}
	}
	r.done.Wait()
}
