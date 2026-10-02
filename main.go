package main

import (
	"bytes"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font/gofont/goregular"
)

type HistoryPoint struct {
	Epoch         int
	TrainAccuracy float32
	TestAccuracy  float32
}

type Game struct {
	screenWidth      int
	screenHeight     int
	currentTab       TabType
	pixelImage       *ebiten.Image
	pixelBuffer      []byte
	points           []DataPoint
	allPoints        []DataPoint
	trainPoints      []DataPoint
	testPoints       []DataPoint
	nn               *NeuralNetwork
	layerSizes       []int
	paramDials       []Slider
	activeSlider     int
	fontFace         *text.GoTextFace
	cost             float32
	accuracy         float32
	trainAccuracy    float32
	testAccuracy     float32
	lr               float32
	batchSize        int
	tickCount        int
	isPaused         bool
	isLearning       bool
	isKilled         bool
	totalEpochs      int
	history          []HistoryPoint
	sliderScrollY    float32
	featureMeans     []float32
	boundaryInput    []float32
	boundaryRenderer *BoundaryRenderer

	// Tab 2 Dialog Box fields
	trainSplitPercent   int
	inputLayerSizes     string
	inputLearnRate      string
	inputBatchSize      string
	inputMomentum       string
	dropdownActivation  *Dropdown
	dropdownOutput      *Dropdown
	dropdownCost        *Dropdown
	activeDropdownId    int // 0: none, 1: activation, 2: output, 3: cost
	activeInputId       int // 0: none, 1: layerSizes, 2: learnRate, 3: batchSize, 4: momentum
	dialogSplitDragging bool
	configError         string
	cursorTick          int

	// Tab 3 MNIST Digit Viewer fields
	mnistDataset    []MNISTImage
	viewerIndex     int
	viewerStatusMsg string
	viewerImage     *ebiten.Image

	// Tab 4 Doodle Canvas fields
	drawPixels      []float32
	drawImage       *ebiten.Image
	drawPrevX       float32
	drawPrevY       float32
	isDrawingActive bool
}

func (g *Game) ToggleLearning() {
	g.isLearning = !g.isLearning
	if g.isLearning {
		g.isKilled = false
	}
}

func (g *Game) KillNetwork() {
	g.isLearning = false
	g.isKilled = true
	g.totalEpochs = 0
	actType := ActivationLeakyReLU
	outType := OutputSoftmax
	costType := CostCrossEntropy
	mom := float32(0.9)
	if g.dropdownActivation != nil {
		actType = ActivationType(g.dropdownActivation.SelectedIndex)
	}
	if g.dropdownOutput != nil {
		outType = OutputType(g.dropdownOutput.SelectedIndex)
	}
	if g.dropdownCost != nil {
		costType = CostType(g.dropdownCost.SelectedIndex)
	}
	if mom64, err := strconv.ParseFloat(strings.TrimSpace(g.inputMomentum), 32); err == nil && mom64 >= 0 && mom64 < 1.0 {
		mom = float32(mom64)
	}
	g.nn = NewConfiguredNeuralNetwork(g.layerSizes, actType, outType, costType, mom)
	g.paramDials = ParseNN(g.nn)
	numFeatures := len(g.points[0].inputs)
	g.boundaryRenderer = NewBoundaryRenderer(g.nn, numFeatures, BOUNDARY_RESOLUTION)
	g.boundaryRenderer.SetFeatureMeans(g.featureMeans)

	trainSet := g.trainPoints
	if len(trainSet) == 0 {
		trainSet = g.points
	}
	testSet := g.testPoints
	if len(testSet) == 0 {
		testSet = trainSet
	}

	g.cost = NetworkCost(g.nn, trainSet)
	g.trainAccuracy = CalculateAccuracy(g.nn, trainSet)
	g.testAccuracy = CalculateAccuracy(g.nn, testSet)
	g.accuracy = g.testAccuracy
	g.history = []HistoryPoint{{Epoch: 0, TrainAccuracy: g.trainAccuracy, TestAccuracy: g.testAccuracy}}
	UpdateSliders(g)
	PlotDecisionBoundary(g)
}

func (g *Game) TotalViewerImages() int {
	if len(g.mnistDataset) > 0 {
		return len(g.mnistDataset)
	}
	return len(g.allPoints)
}

func (g *Game) GetViewerSample(index int) (pixels []float32, trueLabel int, rows, cols int) {
	if len(g.mnistDataset) > 0 {
		if index < 0 {
			index = 0
		}
		if index >= len(g.mnistDataset) {
			index = len(g.mnistDataset) - 1
		}
		img := g.mnistDataset[index]
		r, c := img.Rows, img.Cols
		if r <= 0 || c <= 0 {
			r, c = 28, 28
		}
		return img.Pixels, img.Label, r, c
	}
	if len(g.allPoints) > 0 {
		if index < 0 {
			index = 0
		}
		if index >= len(g.allPoints) {
			index = len(g.allPoints) - 1
		}
		pt := g.allPoints[index]
		label := IndexOfMaxValue(pt.outputs)
		return pt.inputs, label, 28, 28
	}
	return nil, 0, 28, 28
}

func (g *Game) SetViewerImage(index int) {
	total := g.TotalViewerImages()
	if total == 0 {
		return
	}
	if index < 0 {
		index = 0
	}
	if index >= total {
		index = total - 1
	}
	g.viewerIndex = index
	pixels, _, rows, cols := g.GetViewerSample(index)
	if len(pixels) == 0 {
		return
	}

	if g.viewerImage == nil || g.viewerImage.Bounds().Dx() != cols || g.viewerImage.Bounds().Dy() != rows {
		g.viewerImage = ebiten.NewImage(cols, rows)
	}

	pixBytes := make([]byte, rows*cols*4)
	for i, p := range pixels {
		if i >= rows*cols {
			break
		}
		var val byte
		if p > 1.0 {
			val = byte(p)
		} else if p < 0.0 {
			val = 0
		} else {
			val = byte(p * 255.0)
		}
		offset := i * 4
		pixBytes[offset] = val
		pixBytes[offset+1] = val
		pixBytes[offset+2] = val
		pixBytes[offset+3] = 255
	}
	g.viewerImage.WritePixels(pixBytes)
}

func (g *Game) NextViewerImage() {
	total := g.TotalViewerImages()
	if total == 0 {
		return
	}
	g.SetViewerImage((g.viewerIndex + 1) % total)
	g.viewerStatusMsg = ""
}

func (g *Game) PrevViewerImage() {
	total := g.TotalViewerImages()
	if total == 0 {
		return
	}
	g.SetViewerImage((g.viewerIndex - 1 + total) % total)
	g.viewerStatusMsg = ""
}

func (g *Game) RandomViewerImage() {
	total := g.TotalViewerImages()
	if total == 0 {
		return
	}
	g.SetViewerImage(rand.Intn(total))
	g.viewerStatusMsg = ""
}

func (g *Game) getViewerNNInputs(pixels []float32) []float32 {
	if len(pixels) == 0 || g.nn == nil || len(g.layerSizes) == 0 {
		return nil
	}
	inputSize := g.layerSizes[0]
	norm := make([]float32, len(pixels))
	for i, p := range pixels {
		if p > 1.0 {
			norm[i] = p / 255.0
		} else if p < 0.0 {
			norm[i] = 0.0
		} else {
			norm[i] = p
		}
	}
	if len(norm) == inputSize {
		return norm
	}
	pt := DataPoint{inputs: norm}
	return AdjustDataPoints([]DataPoint{pt}, inputSize)[0].inputs
}

func (g *Game) FindNextMisclassified() {
	total := g.TotalViewerImages()
	if total == 0 || g.nn == nil {
		g.viewerStatusMsg = "No images or network available"
		return
	}

	startIdx := (g.viewerIndex + 1) % total

	for step := 0; step < total; step++ {
		idx := (startIdx + step) % total
		pixels, trueLabel, _, _ := g.GetViewerSample(idx)
		if len(pixels) == 0 {
			continue
		}

		inputs := g.getViewerNNInputs(pixels)
		if len(inputs) == 0 {
			continue
		}

		pred := Classify(g.nn, inputs)
		if pred != trueLabel {
			g.SetViewerImage(idx)
			g.viewerStatusMsg = fmt.Sprintf("Found misclassified #%d: True=%d, Pred=%d", idx+1, trueLabel, pred)
			return
		}
	}

	g.viewerStatusMsg = "All images in dataset are correctly classified (100% accuracy)!"
}

func (g *Game) ClearCanvas() {
	if len(g.drawPixels) != 784 {
		g.drawPixels = make([]float32, 784)
	} else {
		for i := range g.drawPixels {
			g.drawPixels[i] = 0
		}
	}
	g.updateCanvasImage()
}

func (g *Game) updateCanvasImage() {
	if g.drawImage == nil {
		g.drawImage = ebiten.NewImage(28, 28)
	}
	if len(g.drawPixels) != 784 {
		return
	}
	pixBytes := make([]byte, 28*28*4)
	for i, p := range g.drawPixels {
		var val byte
		if p >= 1.0 {
			val = 255
		} else if p <= 0.0 {
			val = 0
		} else {
			val = byte(p * 255.0)
		}
		offset := i * 4
		pixBytes[offset] = val
		pixBytes[offset+1] = val
		pixBytes[offset+2] = val
		pixBytes[offset+3] = 255
	}
	g.drawImage.WritePixels(pixBytes)
}

func (g *Game) isCanvasEmpty() bool {
	if len(g.drawPixels) == 0 {
		return true
	}
	sum := float32(0)
	for _, v := range g.drawPixels {
		sum += v
		if sum > 0.5 {
			return false
		}
	}
	return true
}

func (g *Game) PaintCanvasPoint(gx, gy float32) {
	if len(g.drawPixels) != 784 {
		g.drawPixels = make([]float32, 784)
	}
	const radius = float32(1.35)
	minX := int(math.Floor(float64(gx - radius)))
	maxX := int(math.Ceil(float64(gx + radius)))
	minY := int(math.Floor(float64(gy - radius)))
	maxY := int(math.Ceil(float64(gy + radius)))

	if minX < 0 {
		minX = 0
	}
	if maxX > 27 {
		maxX = 27
	}
	if minY < 0 {
		minY = 0
	}
	if maxY > 27 {
		maxY = 27
	}

	for cy := minY; cy <= maxY; cy++ {
		for cx := minX; cx <= maxX; cx++ {
			dx := float32(cx) + 0.5 - gx
			dy := float32(cy) + 0.5 - gy
			dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
			if dist <= radius {
				intensity := 1.0 - (dist/radius)*0.4
				idx := cy*28 + cx
				cur := g.drawPixels[idx]
				newVal := cur + intensity*0.85
				if newVal > 1.0 {
					newVal = 1.0
				}
				g.drawPixels[idx] = newVal
			}
		}
	}
}

func (g *Game) PaintCanvasStroke(x0, y0, x1, y1 float32) {
	dx := x1 - x0
	dy := y1 - y0
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	steps := int(dist/0.4) + 1
	for s := 0; s <= steps; s++ {
		t := float32(s) / float32(steps)
		px := x0 + t*dx
		py := y0 + t*dy
		g.PaintCanvasPoint(px, py)
	}
	g.updateCanvasImage()
}

func (g *Game) RunNewNetwork() error {
	if !g.isKilled {
		g.configError = "Tab 1 network must be killed first!"
		return fmt.Errorf("network in tab 1 must be killed first")
	}

	numOutputs := len(g.allPoints[0].outputs)

	layers, err := ParseLayerSizes(g.inputLayerSizes, numOutputs)
	if err != nil {
		g.configError = err.Error()
		return err
	}
	inputSize := layers[0]

	lr64, err := strconv.ParseFloat(strings.TrimSpace(g.inputLearnRate), 32)
	if err != nil || lr64 <= 0 || lr64 > 10.0 {
		g.configError = "Invalid learn rate (0 < lr <= 10)"
		return fmt.Errorf("invalid learn rate")
	}
	newLR := float32(lr64)

	bs, err := strconv.Atoi(strings.TrimSpace(g.inputBatchSize))
	if err != nil || bs <= 0 {
		g.configError = "Batch size must be > 0"
		return fmt.Errorf("invalid batch size")
	}

	mom := float32(0.9)
	if strings.TrimSpace(g.inputMomentum) != "" {
		mom64, err := strconv.ParseFloat(strings.TrimSpace(g.inputMomentum), 32)
		if err != nil || mom64 < 0 || mom64 >= 1.0 {
			g.configError = "Momentum must be between 0.0 and 0.99"
			return fmt.Errorf("invalid momentum")
		}
		mom = float32(mom64)
	}

	actType := ActivationLeakyReLU
	outType := OutputSoftmax
	costType := CostCrossEntropy
	if g.dropdownActivation != nil {
		actType = ActivationType(g.dropdownActivation.SelectedIndex)
	}
	if g.dropdownOutput != nil {
		outType = OutputType(g.dropdownOutput.SelectedIndex)
	}
	if g.dropdownCost != nil {
		costType = CostType(g.dropdownCost.SelectedIndex)
	}

	adjustedAll := AdjustDataPoints(g.allPoints, inputSize)
	g.trainPoints, g.testPoints = SplitDataset(adjustedAll, g.trainSplitPercent)
	if bs > len(g.trainPoints) {
		bs = len(g.trainPoints)
	}

	g.configError = ""
	g.layerSizes = layers
	g.lr = newLR
	g.batchSize = bs
	g.points = g.trainPoints

	g.nn = NewConfiguredNeuralNetwork(g.layerSizes, actType, outType, costType, mom)
	g.paramDials = ParseNN(g.nn)

	// Recompute feature means and boundary input for the new input size
	g.featureMeans = make([]float32, inputSize)
	for _, pt := range g.trainPoints {
		for f := 0; f < inputSize; f++ {
			g.featureMeans[f] += pt.inputs[f]
		}
	}
	for f := 0; f < inputSize; f++ {
		g.featureMeans[f] /= float32(len(g.trainPoints))
	}
	g.boundaryInput = make([]float32, inputSize)
	copy(g.boundaryInput, g.featureMeans)

	g.boundaryRenderer = NewBoundaryRenderer(g.nn, inputSize, BOUNDARY_RESOLUTION)
	g.boundaryRenderer.SetFeatureMeans(g.featureMeans)

	g.cost = NetworkCost(g.nn, g.trainPoints)
	g.trainAccuracy = CalculateAccuracy(g.nn, g.trainPoints)
	if len(g.testPoints) > 0 {
		g.testAccuracy = CalculateAccuracy(g.nn, g.testPoints)
	} else {
		g.testAccuracy = g.trainAccuracy
	}
	g.accuracy = g.testAccuracy

	g.totalEpochs = 0
	g.history = []HistoryPoint{{Epoch: 0, TrainAccuracy: g.trainAccuracy, TestAccuracy: g.testAccuracy}}
	g.isKilled = false
	g.isLearning = true
	g.isPaused = false

	UpdateSliders(g)
	PlotDecisionBoundary(g)
	return nil
}

func (g *Game) RecordHistory() {
	if len(g.history) > 0 && g.history[len(g.history)-1].Epoch == g.totalEpochs {
		g.history[len(g.history)-1].TrainAccuracy = g.trainAccuracy
		g.history[len(g.history)-1].TestAccuracy = g.testAccuracy
		return
	}
	g.history = append(g.history, HistoryPoint{
		Epoch:         g.totalEpochs,
		TrainAccuracy: g.trainAccuracy,
		TestAccuracy:  g.testAccuracy,
	})
}

func (g *Game) Update() error {
	if inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		g.isPaused = !g.isPaused
	}
	stepRequested := g.isPaused && inpututil.IsKeyJustPressed(ebiten.KeyS)

	x, y := ebiten.CursorPosition()
	mx, my := float32(x), float32(y)
	buttonWidth := float32(18)

	// Tab switching check (mouse click or 1/2/3/4 keys when not typing)
	if g.activeInputId == 0 {
		if inpututil.IsKeyJustPressed(ebiten.Key1) {
			g.currentTab = TabDecisionBoundary
			UpdateSliders(g)
			PlotDecisionBoundary(g)
		} else if inpututil.IsKeyJustPressed(ebiten.Key2) {
			g.currentTab = TabNewFeature
		} else if inpututil.IsKeyJustPressed(ebiten.Key3) {
			g.currentTab = TabDigitViewer
			if g.viewerImage == nil {
				g.SetViewerImage(g.viewerIndex)
			}
		} else if inpututil.IsKeyJustPressed(ebiten.Key4) {
			g.currentTab = TabCanvasDraw
			if g.drawImage == nil {
				g.ClearCanvas()
			}
		}
	}

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) && my < TAB_BAR_HEIGHT {
		clickedTab := int(mx / TAB_WIDTH)
		if clickedTab >= 0 && clickedTab < 4 {
			targetTab := TabType(clickedTab)
			if targetTab == TabDecisionBoundary && g.currentTab != TabDecisionBoundary {
				UpdateSliders(g)
				PlotDecisionBoundary(g)
			}
			if targetTab == TabDigitViewer && g.viewerImage == nil {
				g.SetViewerImage(g.viewerIndex)
			}
			if targetTab == TabCanvasDraw && g.drawImage == nil {
				g.ClearCanvas()
			}
			g.currentTab = targetTab
		}
	}

	// Tab-specific interactions for Tab 1 (Decision Boundary)
	if g.currentTab == TabDecisionBoundary {
		// Mouse wheel scrolling in slider panel
		_, wy := ebiten.Wheel()
		if wy != 0 && mx >= float32(g.screenWidth)-SLIDER_PANEL_WIDTH && my >= TAB_BAR_HEIGHT {
			g.sliderScrollY += float32(wy) * 45.0
		}
		panelH := float32(g.screenHeight) - TAB_BAR_HEIGHT
		totalSliderHeight := float32(len(g.paramDials))*SLIDER_SPACING + 70.0
		maxScroll := totalSliderHeight - (panelH - TOP_BAR_HEIGHT)
		if maxScroll < 0 {
			maxScroll = 0
		}
		if g.sliderScrollY > 0 {
			g.sliderScrollY = 0
		}
		if g.sliderScrollY < -maxScroll {
			g.sliderScrollY = -maxScroll
		}

		// Click on HUD buttons (Learn & Kill)
		if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) && my >= TAB_BAR_HEIGHT && my < TAB_BAR_HEIGHT+TOP_BAR_HEIGHT {
			btnY := TAB_BAR_HEIGHT + 7
			btnH := float32(26)
			// Learn button: x: 12..87
			if mx >= 12 && mx <= 12+75 && my >= btnY && my <= btnY+btnH {
				g.ToggleLearning()
			}
			// Kill button: x: 95..155
			if mx >= 95 && mx <= 95+60 && my >= btnY && my <= btnY+btnH {
				g.KillNetwork()
			}
		}

		if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) && my >= TAB_BAR_HEIGHT+TOP_BAR_HEIGHT {
			for i := range g.paramDials {
				trackX, trackY, trackW, trackH := GetSliderLayout(g.screenWidth, g.screenHeight, i, g.sliderScrollY)
				if mx >= trackX-5 && mx <= trackX+trackW+5 && my >= trackY-8 && my <= trackY+trackH+8 {
					g.activeSlider = i
					s := &g.paramDials[i]
					ratio := (mx - trackX - buttonWidth/2) / (trackW - buttonWidth)
					if ratio < 0 {
						ratio = 0
					}
					if ratio > 1 {
						ratio = 1
					}
					g.paramDials[i].normValue = ratio
					val := g.paramDials[i].Value()
					if s.name == "Weight" {
						g.nn.layers[s.l].weights[s.wi][s.wj] = val
					} else {
						g.nn.layers[s.l].biases[s.b] = val
					}
					PlotDecisionBoundary(g)
					g.cost = NetworkCost(g.nn, g.points)
					g.accuracy = CalculateAccuracy(g.nn, g.points)
					break
				}
			}
		}

		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) && g.activeSlider != -1 {
			trackX, _, trackW, _ := GetSliderLayout(g.screenWidth, g.screenHeight, g.activeSlider, g.sliderScrollY)
			ratio := (mx - trackX - buttonWidth/2) / (trackW - buttonWidth)
			if ratio < 0 {
				ratio = 0
			}
			if ratio > 1 {
				ratio = 1
			}
			if g.paramDials[g.activeSlider].normValue != ratio {
				g.paramDials[g.activeSlider].normValue = ratio
				val := g.paramDials[g.activeSlider].Value()
				s := &g.paramDials[g.activeSlider]
				if s.name == "Weight" {
					g.nn.layers[s.l].weights[s.wi][s.wj] = val
				} else {
					g.nn.layers[s.l].biases[s.b] = val
				}
				PlotDecisionBoundary(g)
				trainSet := g.trainPoints
				if len(trainSet) == 0 {
					trainSet = g.points
				}
				testSet := g.testPoints
				if len(testSet) == 0 {
					testSet = trainSet
				}
				g.cost = NetworkCost(g.nn, trainSet)
				g.trainAccuracy = CalculateAccuracy(g.nn, trainSet)
				g.testAccuracy = CalculateAccuracy(g.nn, testSet)
				g.accuracy = g.testAccuracy
			}
		}

		if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
			g.activeSlider = -1
		}
	}

	if g.currentTab == TabNewFeature {
		g.updateNewFeatureTab(mx, my)
	} else if g.currentTab == TabDigitViewer {
		g.updateDigitViewerTab(mx, my)
	} else if g.currentTab == TabCanvasDraw {
		g.updateCanvasTab(mx, my)
	}

	if g.isLearning && (!g.isPaused || stepRequested) {
		pts := g.trainPoints
		if len(pts) == 0 {
			pts = g.points
		}
		if len(pts) == 0 {
			pts = g.allPoints
		}
		ShuffleDataPoints(pts)
		batchSize := g.batchSize
		if batchSize <= 0 {
			batchSize = 32
		}
		numPoints := len(pts)
		if numPoints > 0 {
			epochs := EPOCHS_PER_FRAME
			if stepRequested {
				epochs = 1
			}
			for j := 0; j < epochs; j += 1 {
				for i := 0; i < numPoints; i += batchSize {
					numEnd := i + batchSize
					if numEnd > numPoints {
						numEnd = numPoints
					}
					Learn(g.nn, pts[i:numEnd], g.lr)
				}
			}
			g.totalEpochs += epochs
		}
		minLR := float32(0.0001)
		if g.lr > minLR {
			g.lr -= 0.00000001
			if g.lr < minLR {
				g.lr = minLR
			}
		}
		trainSet := pts
		testSet := g.testPoints
		if len(testSet) == 0 {
			testSet = trainSet
		}
		if stepRequested {
			if g.currentTab == TabDecisionBoundary {
				UpdateSliders(g)
				PlotDecisionBoundary(g)
			}
			g.cost = NetworkCost(g.nn, trainSet)
			g.trainAccuracy = CalculateAccuracy(g.nn, trainSet)
			g.testAccuracy = CalculateAccuracy(g.nn, testSet)
			g.accuracy = g.testAccuracy
			g.RecordHistory()
		} else {
			g.tickCount++
			if g.tickCount%BOUNDARY_UPDATE_INTERVAL == 0 {
				if g.currentTab == TabDecisionBoundary {
					UpdateSliders(g)
					PlotDecisionBoundary(g)
				}
				g.cost = NetworkCost(g.nn, trainSet)
				g.trainAccuracy = CalculateAccuracy(g.nn, trainSet)
				g.testAccuracy = CalculateAccuracy(g.nn, testSet)
				g.accuracy = g.testAccuracy
				g.RecordHistory()
			}
		}
	}
	return nil
}

func (g *Game) updateNewFeatureTab(mx, my float32) {
	g.cursorTick++

	dialogX := float32(20)
	dialogY := TAB_BAR_HEIGHT + TOP_BAR_HEIGHT + 15
	dialogW := float32(380)
	dialogH := float32(g.screenHeight) - dialogY - 20

	fullW := dialogW - 30
	col1X := dialogX + 15
	colW := (dialogW - 40) / 2
	col2X := col1X + colW + 10
	boxH := float32(28)

	// Row 1: Slider
	trackX := dialogX + 15
	trackY := dialogY + 92
	trackW := fullW
	trackH := float32(10)

	// Row 2: Layer Sizes
	boxLayersX := col1X
	boxLayersY := dialogY + 132

	// Row 3: Architecture (Hidden Activation & Output Function)
	dropActX := col1X
	dropActY := dialogY + 186
	dropOutX := col2X
	dropOutY := dialogY + 186

	// Row 4: Loss & Momentum (Cost Function & Momentum)
	dropCostX := col1X
	dropCostY := dialogY + 240
	boxMomX := col2X
	boxMomY := dialogY + 240

	// Row 5: Optimization (Learning Rate & Batch Size)
	boxLRX := col1X
	boxLRY := dialogY + 294
	boxBSX := col2X
	boxBSY := dialogY + 294

	// Row 6: RUN Button
	btnRunX := dialogX + 15
	btnRunY := dialogY + 352
	btnRunW := fullW
	btnRunH := float32(36)

	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		g.dialogSplitDragging = false
	}

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		// 1. If a dropdown popup is currently open, check if an option was clicked
		if g.activeDropdownId != 0 {
			var d *Dropdown
			var popX, popY, popW float32
			switch g.activeDropdownId {
			case 1:
				d = g.dropdownActivation
				popX, popY, popW = dropActX, dropActY+boxH+2, colW
			case 2:
				d = g.dropdownOutput
				popX, popY, popW = dropOutX, dropOutY+boxH+2, colW
			case 3:
				d = g.dropdownCost
				popX, popY, popW = dropCostX, dropCostY+boxH+2, colW
			}
			if d != nil {
				itemH := float32(26)
				totalH := float32(len(d.Options)) * itemH
				if mx >= popX && mx <= popX+popW && my >= popY && my <= popY+totalH {
					optIdx := int((my - popY) / itemH)
					if optIdx >= 0 && optIdx < len(d.Options) {
						d.SelectedIndex = optIdx
					}
					g.activeDropdownId = 0
					return
				}
			}
			// Clicked outside popup -> close dropdown
			g.activeDropdownId = 0
		}

		// 2. Check dropdown box headers
		if mx >= dropActX && mx <= dropActX+colW && my >= dropActY && my <= dropActY+boxH {
			g.activeDropdownId = 1
			g.activeInputId = 0
			return
		} else if mx >= dropOutX && mx <= dropOutX+colW && my >= dropOutY && my <= dropOutY+boxH {
			g.activeDropdownId = 2
			g.activeInputId = 0
			return
		} else if mx >= dropCostX && mx <= dropCostX+colW && my >= dropCostY && my <= dropCostY+boxH {
			g.activeDropdownId = 3
			g.activeInputId = 0
			return
		}

		// 3. Check slider
		if mx >= trackX-5 && mx <= trackX+trackW+5 && my >= trackY-8 && my <= trackY+trackH+8 {
			g.dialogSplitDragging = true
			ratio := (mx - trackX) / trackW
			if ratio < 0 {
				ratio = 0
			}
			if ratio > 1 {
				ratio = 1
			}
			g.trainSplitPercent = 10 + int(ratio*80.0)
			return
		}

		// 4. Check text input fields
		if mx >= boxLayersX && mx <= boxLayersX+fullW && my >= boxLayersY && my <= boxLayersY+boxH {
			g.activeInputId = 1
		} else if mx >= boxLRX && mx <= boxLRX+colW && my >= boxLRY && my <= boxLRY+boxH {
			g.activeInputId = 2
		} else if mx >= boxMomX && mx <= boxMomX+colW && my >= boxMomY && my <= boxMomY+boxH {
			g.activeInputId = 4
		} else if mx >= boxBSX && mx <= boxBSX+colW && my >= boxBSY && my <= boxBSY+boxH {
			g.activeInputId = 3
		} else if mx >= btnRunX && mx <= btnRunX+btnRunW && my >= btnRunY && my <= btnRunY+btnRunH {
			if g.isKilled {
				g.RunNewNetwork()
			} else {
				g.configError = "Kill Tab 1 network first!"
			}
		} else if mx >= dialogX && mx <= dialogX+dialogW && my >= dialogY && my <= dialogY+dialogH {
			g.activeInputId = 0
		}
	}

	if g.dialogSplitDragging && ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		ratio := (mx - trackX) / trackW
		if ratio < 0 {
			ratio = 0
		}
		if ratio > 1 {
			ratio = 1
		}
		g.trainSplitPercent = 10 + int(ratio*80.0)
	}

	if g.activeInputId != 0 {
		if inpututil.IsKeyJustPressed(ebiten.KeyTab) {
			switch g.activeInputId {
			case 1:
				g.activeInputId = 2
			case 2:
				g.activeInputId = 4
			case 4:
				g.activeInputId = 3
			case 3:
				g.activeInputId = 1
			default:
				g.activeInputId = 1
			}
		}

		if inpututil.IsKeyJustPressed(ebiten.KeyEnter) || inpututil.IsKeyJustPressed(ebiten.KeyNumpadEnter) {
			if g.isKilled {
				g.RunNewNetwork()
			} else {
				g.configError = "Kill Tab 1 network first!"
			}
		}

		backspacePressed := inpututil.IsKeyJustPressed(ebiten.KeyBackspace) ||
			(inpututil.KeyPressDuration(ebiten.KeyBackspace) > 30 && inpututil.KeyPressDuration(ebiten.KeyBackspace)%4 == 0)

		var targetStr *string
		switch g.activeInputId {
		case 1:
			targetStr = &g.inputLayerSizes
		case 2:
			targetStr = &g.inputLearnRate
		case 3:
			targetStr = &g.inputBatchSize
		case 4:
			targetStr = &g.inputMomentum
		}

		if targetStr != nil {
			if backspacePressed && len(*targetStr) > 0 {
				*targetStr = (*targetStr)[:len(*targetStr)-1]
				g.configError = ""
			}

			chars := ebiten.AppendInputChars(nil)
			for _, c := range chars {
				if (c >= '0' && c <= '9') || c == ',' || c == '.' || c == ' ' {
					if len(*targetStr) < 60 {
						*targetStr += string(c)
						g.configError = ""
					}
				}
			}
		}
	}
}

func UpdateSliders(g *Game) {
	for i := range g.paramDials {
		if i == g.activeSlider {
			continue
		}
		s := &g.paramDials[i]
		if s.name == "Weight" {
			s.SetValue(g.nn.layers[s.l].weights[s.wi][s.wj])
		} else {
			s.SetValue(g.nn.layers[s.l].biases[s.b])
		}
	}
}

func PlotDecisionBoundary(g *Game) {
	if len(g.points) == 0 {
		return
	}
	if len(g.boundaryInput) == 0 {
		numFeatures := len(g.points[0].inputs)
		g.boundaryInput = make([]float32, numFeatures)
		g.featureMeans = make([]float32, numFeatures)
		for _, pt := range g.points {
			for f := 0; f < numFeatures; f++ {
				g.featureMeans[f] += pt.inputs[f]
			}
		}
		for f := 0; f < numFeatures; f++ {
			g.featureMeans[f] /= float32(len(g.points))
		}
	}

	if g.boundaryRenderer == nil {
		numFeatures := len(g.points[0].inputs)
		g.boundaryRenderer = NewBoundaryRenderer(g.nn, numFeatures, BOUNDARY_RESOLUTION)
	}

	if len(g.pixelBuffer) < BOUNDARY_RESOLUTION*BOUNDARY_RESOLUTION*4 {
		g.pixelBuffer = make([]byte, BOUNDARY_RESOLUTION*BOUNDARY_RESOLUTION*4)
	}
	g.boundaryRenderer.SetFeatureMeans(g.featureMeans)
	g.boundaryRenderer.Render(g.pixelBuffer)

	if g.pixelImage != nil {
		g.pixelImage.WritePixels(g.pixelBuffer)
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(BG_COLOR)

	DrawTabBar(g, screen)

	switch g.currentTab {
	case TabDecisionBoundary:
		g.drawDecisionBoundaryTab(screen)
	case TabNewFeature:
		g.drawNewFeatureTab(screen)
	case TabDigitViewer:
		g.drawDigitViewerTab(screen)
	case TabCanvasDraw:
		g.drawCanvasTab(screen)
	}
}

func DrawTabBar(g *Game, screen *ebiten.Image) {
	width := float32(g.screenWidth)
	vector.FillRect(screen, 0, 0, width, TAB_BAR_HEIGHT, COLOR_4, true)
	vector.StrokeLine(screen, 0, TAB_BAR_HEIGHT, width, TAB_BAR_HEIGHT, 1, LINE_COLOR, true)

	tabTitles := []string{"Decision Boundary", "New Feature", "MNIST Inspector", "Doodle Canvas"}
	mx, my := ebiten.CursorPosition()
	fmx, fmy := float32(mx), float32(my)

	for i, title := range tabTitles {
		tX := float32(i) * TAB_WIDTH
		isActive := g.currentTab == TabType(i)
		isHovered := !isActive && fmx >= tX && fmx < tX+TAB_WIDTH && fmy >= 0 && fmy < TAB_BAR_HEIGHT

		if isActive {
			vector.FillRect(screen, tX, 0, TAB_WIDTH, TAB_BAR_HEIGHT, BG_COLOR, true)
			vector.FillRect(screen, tX, TAB_BAR_HEIGHT-3, TAB_WIDTH, 3, COLOR_1, true)
		} else if isHovered {
			vector.FillRect(screen, tX, 0, TAB_WIDTH, TAB_BAR_HEIGHT, color.RGBA{R: 28, G: 30, B: 34, A: 255}, true)
		}

		vector.StrokeLine(screen, tX+TAB_WIDTH, 0, tX+TAB_WIDTH, TAB_BAR_HEIGHT, 1, LINE_COLOR, true)

		op := &text.DrawOptions{}
		op.GeoM.Translate(float64(tX+18), 10)
		if isActive {
			op.ColorScale.ScaleWithColor(COLOR_1)
		} else if isHovered {
			op.ColorScale.ScaleWithColor(color.RGBA{R: 210, G: 210, B: 220, A: 255})
		} else {
			op.ColorScale.ScaleWithColor(COLOR_3)
		}
		text.Draw(screen, fmt.Sprintf("%d. %s", i+1, title), g.fontFace, op)
	}
}

func (g *Game) drawDecisionBoundaryTab(screen *ebiten.Image) {
	graphW := float32(g.screenWidth) - SLIDER_PANEL_WIDTH
	graphH := float32(g.screenHeight)
	marginX := float32(10)
	marginY := float32(10)
	plotX := marginX
	plotY := TAB_BAR_HEIGHT + TOP_BAR_HEIGHT + marginY
	plotW := graphW - marginX*2
	plotH := graphH - plotY - marginY

	if plotW > 0 && plotH > 0 {
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Scale(float64(plotW)/float64(BOUNDARY_RESOLUTION), float64(plotH)/float64(BOUNDARY_RESOLUTION))
		op.GeoM.Translate(float64(plotX), float64(plotY))
		screen.DrawImage(g.pixelImage, op)
		DrawGrid(screen, plotX, plotY, plotW, plotH)
		PlotPoints(g, screen, plotX, plotY, plotW, plotH)
	}

	DrawSliders(g, screen)
	DrawHUD(g, screen, graphW)
}

func (g *Game) drawNewFeatureTab(screen *ebiten.Image) {
	contentY := TAB_BAR_HEIGHT
	contentW := float32(g.screenWidth)

	bannerH := TOP_BAR_HEIGHT
	vector.FillRect(screen, 0, contentY, contentW, bannerH, COLOR_4, true)
	vector.StrokeLine(screen, 0, contentY+bannerH, contentW, contentY+bannerH, 1, LINE_COLOR, true)

	headerOp := &text.DrawOptions{}
	headerOp.GeoM.Translate(15, float64(contentY+12))
	headerOp.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, "TRAINING METRICS & NEW NETWORK", g.fontFace, headerOp)

	infoOp := &text.DrawOptions{}
	infoOp.GeoM.Translate(260, float64(contentY+12))
	infoOp.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, fmt.Sprintf("Train Acc: %.1f%%  |  Test Acc: %.1f%%  |  Total Epochs: %d", g.trainAccuracy, g.testAccuracy, g.totalEpochs), g.fontFace, infoOp)

	// Display learning indicator badge only when previous network is learning
	if g.isLearning {
		badgeW := float32(230)
		badgeH := float32(24)
		badgeX := contentW - badgeW - 20
		badgeY := contentY + 8
		if badgeX > 520 {
			vector.FillRect(screen, badgeX, badgeY, badgeW, badgeH, color.RGBA{R: 20, G: 55, B: 30, A: 255}, true)
			vector.StrokeRect(screen, badgeX, badgeY, badgeW, badgeH, 1, color.RGBA{R: 76, G: 175, B: 80, A: 255}, true)

			bOp := &text.DrawOptions{}
			bOp.GeoM.Translate(float64(badgeX+12), float64(badgeY+5))
			bOp.ColorScale.ScaleWithColor(color.RGBA{R: 120, G: 240, B: 130, A: 255})
			text.Draw(screen, "● previous network is learning", g.fontFace, bOp)
		}
	}

	dialogX := float32(20)
	dialogY := contentY + bannerH + 15
	dialogW := float32(380)
	dialogH := float32(g.screenHeight) - dialogY - 20

	fullW := dialogW - 30
	col1X := dialogX + 15
	colW := (dialogW - 40) / 2
	col2X := col1X + colW + 10
	boxH := float32(28)

	// Row 1: Slider
	trackX := dialogX + 15
	trackY := dialogY + 92
	trackW := fullW
	trackH := float32(10)

	// Row 2: Layer Sizes
	boxLayersX := col1X
	boxLayersY := dialogY + 132

	// Row 3: Architecture (Hidden Activation & Output Function)
	dropActX := col1X
	dropActY := dialogY + 186
	dropOutX := col2X
	dropOutY := dialogY + 186

	// Row 4: Loss & Momentum (Cost Function & Momentum)
	dropCostX := col1X
	dropCostY := dialogY + 240
	boxMomX := col2X
	boxMomY := dialogY + 240

	// Row 5: Optimization (Learning Rate & Batch Size)
	boxLRX := col1X
	boxLRY := dialogY + 294
	boxBSX := col2X
	boxBSY := dialogY + 294

	// Row 6: RUN Button
	btnRunX := dialogX + 15
	btnRunY := dialogY + 352
	btnRunW := fullW
	btnRunH := float32(36)

	// 1. DIALOG BOX PANEL
	if dialogH > 150 {
		vector.FillRect(screen, dialogX, dialogY, dialogW, dialogH, COLOR_4, true)
		vector.StrokeRect(screen, dialogX, dialogY, dialogW, dialogH, 1, LINE_COLOR, true)

		// Title Bar
		titleH := float32(34)
		vector.FillRect(screen, dialogX, dialogY, dialogW, titleH, color.RGBA{R: 28, G: 30, B: 34, A: 255}, true)
		vector.StrokeLine(screen, dialogX, dialogY+titleH, dialogX+dialogW, dialogY+titleH, 1, LINE_COLOR, true)
		dTitleOp := &text.DrawOptions{}
		dTitleOp.GeoM.Translate(float64(dialogX+15), float64(dialogY+10))
		dTitleOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "CONFIGURE NEW NETWORK", g.fontFace, dTitleOp)

		// Status Lock Banner
		bannerBoxW := fullW
		bannerBoxH := float32(24)
		bannerBoxX := dialogX + 15
		bannerBoxY := dialogY + 42
		if !g.isKilled {
			vector.FillRect(screen, bannerBoxX, bannerBoxY, bannerBoxW, bannerBoxH, color.RGBA{R: 50, G: 22, B: 24, A: 255}, true)
			vector.StrokeRect(screen, bannerBoxX, bannerBoxY, bannerBoxW, bannerBoxH, 1, color.RGBA{R: 190, G: 60, B: 60, A: 255}, true)
			bOp := &text.DrawOptions{}
			bOp.GeoM.Translate(float64(bannerBoxX+8), float64(bannerBoxY+5))
			bOp.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 140, B: 140, A: 255})
			text.Draw(screen, "⚠️ LOCKED: Kill Tab 1 network first", g.fontFace, bOp)
		} else {
			vector.FillRect(screen, bannerBoxX, bannerBoxY, bannerBoxW, bannerBoxH, color.RGBA{R: 20, G: 50, B: 28, A: 255}, true)
			vector.StrokeRect(screen, bannerBoxX, bannerBoxY, bannerBoxW, bannerBoxH, 1, color.RGBA{R: 60, G: 170, B: 80, A: 255}, true)
			bOp := &text.DrawOptions{}
			bOp.GeoM.Translate(float64(bannerBoxX+8), float64(bannerBoxY+5))
			bOp.ColorScale.ScaleWithColor(color.RGBA{R: 130, G: 240, B: 150, A: 255})
			text.Draw(screen, "✓ READY: Tab 1 network is killed", g.fontFace, bOp)
		}

		// Row 1: Training Split Slider
		lblOp := &text.DrawOptions{}
		lblOp.GeoM.Translate(float64(dialogX+15), float64(dialogY+72))
		lblOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, fmt.Sprintf("Train Split: %d%%  |  Test: %d%%", g.trainSplitPercent, 100-g.trainSplitPercent), g.fontFace, lblOp)

		vector.FillRect(screen, trackX, trackY+2, trackW, 6, LINE_COLOR, true)
		knobW := float32(18)
		knobX := trackX + (float32(g.trainSplitPercent-10)/80.0)*(trackW-knobW)
		vector.FillRect(screen, knobX, trackY-2, knobW, trackH+4, COLOR_1, true)

		// Row 2: Layer Sizes Input Box
		lbl1Op := &text.DrawOptions{}
		lbl1Op.GeoM.Translate(float64(dialogX+15), float64(dialogY+114))
		lbl1Op.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Layer Sizes (e.g. 784, 128, 64, 10):", g.fontFace, lbl1Op)

		vector.FillRect(screen, boxLayersX, boxLayersY, fullW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		box1Border := LINE_COLOR
		if g.activeInputId == 1 {
			box1Border = COLOR_1
		}
		vector.StrokeRect(screen, boxLayersX, boxLayersY, fullW, boxH, 1, box1Border, true)

		txt1 := g.inputLayerSizes
		if g.activeInputId == 1 && (g.cursorTick/30)%2 == 0 {
			txt1 += "|"
		}
		t1Op := &text.DrawOptions{}
		t1Op.GeoM.Translate(float64(boxLayersX+10), float64(boxLayersY+6))
		t1Op.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, txt1, g.fontFace, t1Op)

		// Row 3 (Architecture):
		// Left: Hidden Activation Dropdown
		lblActOp := &text.DrawOptions{}
		lblActOp.GeoM.Translate(float64(dropActX), float64(dialogY+168))
		lblActOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Hidden Activation:", g.fontFace, lblActOp)

		vector.FillRect(screen, dropActX, dropActY, colW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		actBorder := LINE_COLOR
		if g.activeDropdownId == 1 {
			actBorder = COLOR_1
		}
		vector.StrokeRect(screen, dropActX, dropActY, colW, boxH, 1, actBorder, true)
		actVal := "LeakyReLU"
		if g.dropdownActivation != nil {
			actVal = g.dropdownActivation.SelectedValue()
		}
		actValOp := &text.DrawOptions{}
		actValOp.GeoM.Translate(float64(dropActX+8), float64(dropActY+6))
		actValOp.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, actVal, g.fontFace, actValOp)
		actArrOp := &text.DrawOptions{}
		actArrOp.GeoM.Translate(float64(dropActX+colW-18), float64(dropActY+6))
		actArrOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "v", g.fontFace, actArrOp)

		// Right: Output Function Dropdown
		lblOutOp := &text.DrawOptions{}
		lblOutOp.GeoM.Translate(float64(dropOutX), float64(dialogY+168))
		lblOutOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Output Function:", g.fontFace, lblOutOp)

		vector.FillRect(screen, dropOutX, dropOutY, colW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		outBorder := LINE_COLOR
		if g.activeDropdownId == 2 {
			outBorder = COLOR_1
		}
		vector.StrokeRect(screen, dropOutX, dropOutY, colW, boxH, 1, outBorder, true)
		outVal := "Softmax"
		if g.dropdownOutput != nil {
			outVal = g.dropdownOutput.SelectedValue()
		}
		outValOp := &text.DrawOptions{}
		outValOp.GeoM.Translate(float64(dropOutX+8), float64(dropOutY+6))
		outValOp.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, outVal, g.fontFace, outValOp)
		outArrOp := &text.DrawOptions{}
		outArrOp.GeoM.Translate(float64(dropOutX+colW-18), float64(dropOutY+6))
		outArrOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "v", g.fontFace, outArrOp)

		// Row 4 (Loss & Momentum):
		// Left: Cost Function Dropdown
		lblCostOp := &text.DrawOptions{}
		lblCostOp.GeoM.Translate(float64(dropCostX), float64(dialogY+222))
		lblCostOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Cost Function:", g.fontFace, lblCostOp)

		vector.FillRect(screen, dropCostX, dropCostY, colW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		costBorder := LINE_COLOR
		if g.activeDropdownId == 3 {
			costBorder = COLOR_1
		}
		vector.StrokeRect(screen, dropCostX, dropCostY, colW, boxH, 1, costBorder, true)
		costVal := "Cross-Entropy"
		if g.dropdownCost != nil {
			costVal = g.dropdownCost.SelectedValue()
		}
		costValOp := &text.DrawOptions{}
		costValOp.GeoM.Translate(float64(dropCostX+8), float64(dropCostY+6))
		costValOp.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, costVal, g.fontFace, costValOp)
		costArrOp := &text.DrawOptions{}
		costArrOp.GeoM.Translate(float64(dropCostX+colW-18), float64(dropCostY+6))
		costArrOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "v", g.fontFace, costArrOp)

		// Right: Momentum Input Box
		lblMomOp := &text.DrawOptions{}
		lblMomOp.GeoM.Translate(float64(boxMomX), float64(dialogY+222))
		lblMomOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Momentum (0-0.99):", g.fontFace, lblMomOp)

		vector.FillRect(screen, boxMomX, boxMomY, colW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		boxMomBorder := LINE_COLOR
		if g.activeInputId == 4 {
			boxMomBorder = COLOR_1
		}
		vector.StrokeRect(screen, boxMomX, boxMomY, colW, boxH, 1, boxMomBorder, true)
		txtMom := g.inputMomentum
		if g.activeInputId == 4 && (g.cursorTick/30)%2 == 0 {
			txtMom += "|"
		}
		tmOp := &text.DrawOptions{}
		tmOp.GeoM.Translate(float64(boxMomX+8), float64(boxMomY+6))
		tmOp.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, txtMom, g.fontFace, tmOp)

		// Row 5 (Optimization):
		// Left: Learning Rate Input Box
		lbl2Op := &text.DrawOptions{}
		lbl2Op.GeoM.Translate(float64(boxLRX), float64(dialogY+276))
		lbl2Op.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Learning Rate:", g.fontFace, lbl2Op)

		vector.FillRect(screen, boxLRX, boxLRY, colW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		box2Border := LINE_COLOR
		if g.activeInputId == 2 {
			box2Border = COLOR_1
		}
		vector.StrokeRect(screen, boxLRX, boxLRY, colW, boxH, 1, box2Border, true)
		txt2 := g.inputLearnRate
		if g.activeInputId == 2 && (g.cursorTick/30)%2 == 0 {
			txt2 += "|"
		}
		t2Op := &text.DrawOptions{}
		t2Op.GeoM.Translate(float64(boxLRX+8), float64(boxLRY+6))
		t2Op.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, txt2, g.fontFace, t2Op)

		// Right: Mini Batch Size Input Box
		lbl3Op := &text.DrawOptions{}
		lbl3Op.GeoM.Translate(float64(boxBSX), float64(dialogY+276))
		lbl3Op.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Mini Batch Size:", g.fontFace, lbl3Op)

		vector.FillRect(screen, boxBSX, boxBSY, colW, boxH, color.RGBA{R: 20, G: 22, B: 25, A: 255}, true)
		box3Border := LINE_COLOR
		if g.activeInputId == 3 {
			box3Border = COLOR_1
		}
		vector.StrokeRect(screen, boxBSX, boxBSY, colW, boxH, 1, box3Border, true)
		txt3 := g.inputBatchSize
		if g.activeInputId == 3 && (g.cursorTick/30)%2 == 0 {
			txt3 += "|"
		}
		t3Op := &text.DrawOptions{}
		t3Op.GeoM.Translate(float64(boxBSX+8), float64(boxBSY+6))
		t3Op.ColorScale.ScaleWithColor(color.RGBA{R: 240, G: 240, B: 240, A: 255})
		text.Draw(screen, txt3, g.fontFace, t3Op)

		// Config Error Message
		if g.configError != "" {
			errOp := &text.DrawOptions{}
			errOp.GeoM.Translate(float64(dialogX+15), float64(dialogY+330))
			errOp.ColorScale.ScaleWithColor(COLOR_2)
			text.Draw(screen, g.configError, g.fontFace, errOp)
		}

		// Row 6: RUN Button
		mx, my := ebiten.CursorPosition()
		fmx, fmy := float32(mx), float32(my)
		isRunHover := fmx >= btnRunX && fmx <= btnRunX+btnRunW && fmy >= btnRunY && fmy <= btnRunY+btnRunH

		if g.isKilled {
			runBg := color.RGBA{R: 35, G: 125, B: 50, A: 255}
			if isRunHover {
				runBg = color.RGBA{R: 45, G: 155, B: 65, A: 255}
			}
			vector.FillRect(screen, btnRunX, btnRunY, btnRunW, btnRunH, runBg, true)
			vector.StrokeRect(screen, btnRunX, btnRunY, btnRunW, btnRunH, 1, color.RGBA{R: 90, G: 210, B: 100, A: 255}, true)
			rOp := &text.DrawOptions{}
			rOp.GeoM.Translate(float64(btnRunX+85), float64(btnRunY+10))
			rOp.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 255, B: 255, A: 255})
			text.Draw(screen, "▶  RUN NEW NETWORK", g.fontFace, rOp)
		} else {
			runBg := color.RGBA{R: 32, G: 34, B: 38, A: 255}
			vector.FillRect(screen, btnRunX, btnRunY, btnRunW, btnRunH, runBg, true)
			vector.StrokeRect(screen, btnRunX, btnRunY, btnRunW, btnRunH, 1, LINE_COLOR, true)
			rOp := &text.DrawOptions{}
			rOp.GeoM.Translate(float64(btnRunX+50), float64(btnRunY+10))
			rOp.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, "🔒 LOCKED (KILL TAB 1 FIRST)", g.fontFace, rOp)
		}

		// Hint text
		hintOp := &text.DrawOptions{}
		hintOp.GeoM.Translate(float64(dialogX+15), float64(dialogY+398))
		hintOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Tip: Tab switches fields | Enter runs", g.fontFace, hintOp)

		// FLOATING DROPDOWN POPUP MENU (rendered last so it floats on top!)
		if g.activeDropdownId != 0 {
			var d *Dropdown
			var popX, popY, popW float32
			switch g.activeDropdownId {
			case 1:
				d = g.dropdownActivation
				popX, popY, popW = dropActX, dropActY+boxH+2, colW
			case 2:
				d = g.dropdownOutput
				popX, popY, popW = dropOutX, dropOutY+boxH+2, colW
			case 3:
				d = g.dropdownCost
				popX, popY, popW = dropCostX, dropCostY+boxH+2, colW
			}

			if d != nil {
				itemH := float32(26)
				totalH := float32(len(d.Options)) * itemH
				// Shadow & background card
				vector.FillRect(screen, popX, popY, popW, totalH, color.RGBA{R: 22, G: 25, B: 30, A: 255}, true)
				vector.StrokeRect(screen, popX, popY, popW, totalH, 1.5, COLOR_1, true)

				for i, opt := range d.Options {
					itemY := popY + float32(i)*itemH
					isHover := fmx >= popX && fmx <= popX+popW && fmy >= itemY && fmy < itemY+itemH
					if isHover {
						vector.FillRect(screen, popX+1, itemY+1, popW-2, itemH-2, color.RGBA{R: 45, G: 75, B: 120, A: 255}, true)
					}
					itemOp := &text.DrawOptions{}
					itemOp.GeoM.Translate(float64(popX+10), float64(itemY+5))
					if i == d.SelectedIndex {
						itemOp.ColorScale.ScaleWithColor(color.RGBA{R: 100, G: 220, B: 255, A: 255})
						text.Draw(screen, "• "+opt, g.fontFace, itemOp)
					} else {
						itemOp.ColorScale.ScaleWithColor(color.RGBA{R: 220, G: 220, B: 220, A: 255})
						text.Draw(screen, "  "+opt, g.fontFace, itemOp)
					}
				}
			}
		}
	}

	// 2. ACCURACY VS. EPOCHS GRAPH PANEL (Right side)
	graphCardX := dialogX + dialogW + 15
	graphCardY := dialogY
	graphCardW := contentW - graphCardX - 20
	graphCardH := dialogH

	if graphCardW > 120 && graphCardH > 100 {
		vector.FillRect(screen, graphCardX, graphCardY, graphCardW, graphCardH, COLOR_4, true)
		vector.StrokeRect(screen, graphCardX, graphCardY, graphCardW, graphCardH, 1, LINE_COLOR, true)

		// Card title
		titleOp := &text.DrawOptions{}
		titleOp.GeoM.Translate(float64(graphCardX+20), float64(graphCardY+15))
		titleOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "ACCURACY VS. NUMBER OF EPOCHS", g.fontFace, titleOp)

		testColor := color.RGBA{R: 76, G: 215, B: 100, A: 255}

		// Legend in top-right of graph card
		legendX := graphCardX + graphCardW - 250
		if legendX > graphCardX+280 {
			// Train legend (Blue)
			vector.FillRect(screen, legendX, graphCardY+16, 10, 10, COLOR_1, true)
			trOp := &text.DrawOptions{}
			trOp.GeoM.Translate(float64(legendX+14), float64(graphCardY+14))
			trOp.ColorScale.ScaleWithColor(COLOR_1)
			text.Draw(screen, fmt.Sprintf("Train: %.1f%%", g.trainAccuracy), g.fontFace, trOp)

			// Test legend (Green)
			vector.FillRect(screen, legendX+125, graphCardY+16, 10, 10, testColor, true)
			teOp := &text.DrawOptions{}
			teOp.GeoM.Translate(float64(legendX+139), float64(graphCardY+14))
			teOp.ColorScale.ScaleWithColor(testColor)
			text.Draw(screen, fmt.Sprintf("Test: %.1f%%", g.testAccuracy), g.fontFace, teOp)
		}

		plotLeftMargin := float32(65)
		plotBottomMargin := float32(50)
		plotTopMargin := float32(45)
		plotRightMargin := float32(30)

		plotX := graphCardX + plotLeftMargin
		plotY := graphCardY + plotTopMargin
		plotW := graphCardW - plotLeftMargin - plotRightMargin
		plotH := graphCardH - plotTopMargin - plotBottomMargin

		if plotW > 50 && plotH > 50 {
			// Y-Axis: Accuracy (0% to 100%)
			for k := 0; k <= 4; k++ {
				pct := float32(k) * 25.0
				gy := plotY + plotH - (float32(k)/4.0)*plotH
				vector.StrokeLine(screen, plotX, gy, plotX+plotW, gy, 1, color.RGBA{R: 45, G: 47, B: 50, A: 255}, true)

				yLabelOp := &text.DrawOptions{}
				yLabelOp.GeoM.Translate(float64(plotX-45), float64(gy-7))
				yLabelOp.ColorScale.ScaleWithColor(COLOR_3)
				text.Draw(screen, fmt.Sprintf("%3.0f%%", pct), g.fontFace, yLabelOp)
			}

			// Y-axis title
			yTitleOp := &text.DrawOptions{}
			yTitleOp.GeoM.Translate(float64(plotX-55), float64(plotY-18))
			yTitleOp.ColorScale.ScaleWithColor(COLOR_2)
			text.Draw(screen, "Accuracy", g.fontFace, yTitleOp)

			// X-Axis: Number of Epochs
			maxEpochs := g.totalEpochs
			if maxEpochs < 50 {
				maxEpochs = 50
			}
			for k := 0; k <= 4; k++ {
				gx := plotX + (float32(k)/4.0)*plotW
				vector.StrokeLine(screen, gx, plotY, gx, plotY+plotH, 1, color.RGBA{R: 45, G: 47, B: 50, A: 255}, true)

				epochVal := int(float32(maxEpochs) * float32(k) / 4.0)
				xLabelOp := &text.DrawOptions{}
				xLabelOp.GeoM.Translate(float64(gx-12), float64(plotY+plotH+8))
				xLabelOp.ColorScale.ScaleWithColor(COLOR_3)
				text.Draw(screen, fmt.Sprintf("%d", epochVal), g.fontFace, xLabelOp)
			}

			// X-axis title
			xTitleOp := &text.DrawOptions{}
			xTitleOp.GeoM.Translate(float64(plotX+plotW/2-55), float64(plotY+plotH+28))
			xTitleOp.ColorScale.ScaleWithColor(COLOR_1)
			text.Draw(screen, "Number of Epochs", g.fontFace, xTitleOp)

			// Axes borders
			vector.StrokeLine(screen, plotX, plotY, plotX, plotY+plotH, 2, LINE_COLOR, true)
			vector.StrokeLine(screen, plotX, plotY+plotH, plotX+plotW, plotY+plotH, 2, LINE_COLOR, true)

			// Plot history points for Train and Test accuracies
			if len(g.history) > 1 {
				var lastX, lastTrainY, lastTestY float32
				for i := 1; i < len(g.history); i++ {
					p0 := g.history[i-1]
					p1 := g.history[i]
					x0 := plotX + (float32(p0.Epoch)/float32(maxEpochs))*plotW
					x1 := plotX + (float32(p1.Epoch)/float32(maxEpochs))*plotW

					y0Train := plotY + plotH - (p0.TrainAccuracy/100.0)*plotH
					y1Train := plotY + plotH - (p1.TrainAccuracy/100.0)*plotH
					if y0Train < plotY {
						y0Train = plotY
					}
					if y1Train < plotY {
						y1Train = plotY
					}
					if y0Train > plotY+plotH {
						y0Train = plotY + plotH
					}
					if y1Train > plotY+plotH {
						y1Train = plotY + plotH
					}
					vector.StrokeLine(screen, x0, y0Train, x1, y1Train, 2, COLOR_1, true)

					y0Test := plotY + plotH - (p0.TestAccuracy/100.0)*plotH
					y1Test := plotY + plotH - (p1.TestAccuracy/100.0)*plotH
					if y0Test < plotY {
						y0Test = plotY
					}
					if y1Test < plotY {
						y1Test = plotY
					}
					if y0Test > plotY+plotH {
						y0Test = plotY + plotH
					}
					if y1Test > plotY+plotH {
						y1Test = plotY + plotH
					}
					vector.StrokeLine(screen, x0, y0Test, x1, y1Test, 2, testColor, true)

					lastX = x1
					lastTrainY = y1Train
					lastTestY = y1Test
				}

				// Current point markers
				vector.FillCircle(screen, lastX, lastTrainY, 4, COLOR_1, true)
				vector.FillCircle(screen, lastX, lastTestY, 4, testColor, true)

				// Value labels near the end markers
				trainLblY := lastTrainY - 12
				testLblY := lastTestY - 12
				diff := trainLblY - testLblY
				if diff < 0 {
					diff = -diff
				}
				if diff < 14 {
					if lastTrainY <= lastTestY {
						trainLblY = lastTrainY - 14
						testLblY = lastTestY + 4
					} else {
						trainLblY = lastTrainY + 4
						testLblY = lastTestY - 14
					}
				}

				valTrOp := &text.DrawOptions{}
				valTrOp.GeoM.Translate(float64(lastX+6), float64(trainLblY))
				valTrOp.ColorScale.ScaleWithColor(COLOR_1)
				text.Draw(screen, fmt.Sprintf("%.1f%%", g.trainAccuracy), g.fontFace, valTrOp)

				valTeOp := &text.DrawOptions{}
				valTeOp.GeoM.Translate(float64(lastX+6), float64(testLblY))
				valTeOp.ColorScale.ScaleWithColor(testColor)
				text.Draw(screen, fmt.Sprintf("%.1f%%", g.testAccuracy), g.fontFace, valTeOp)
			} else if len(g.history) == 1 {
				p := g.history[0]
				px := plotX + (float32(p.Epoch)/float32(maxEpochs))*plotW
				pyTrain := plotY + plotH - (p.TrainAccuracy/100.0)*plotH
				pyTest := plotY + plotH - (p.TestAccuracy/100.0)*plotH
				vector.FillCircle(screen, px, pyTrain, 4, COLOR_1, true)
				vector.FillCircle(screen, px, pyTest, 4, testColor, true)
			}

			if !g.isLearning && g.totalEpochs == 0 {
				emptyOp := &text.DrawOptions{}
				emptyOp.GeoM.Translate(float64(plotX+plotW/2-140), float64(plotY+plotH/2-10))
				emptyOp.ColorScale.ScaleWithColor(COLOR_3)
				text.Draw(screen, "Configure parameters and click 'RUN NEW NETWORK'.", g.fontFace, emptyOp)
			}
		}
	}
}

func getNormalizedProbabilities(rawOutputs []float32) []float32 {
	probs := make([]float32, len(rawOutputs))
	if len(rawOutputs) == 0 {
		return probs
	}
	sum := float32(0)
	allNonNeg := true
	for _, v := range rawOutputs {
		if v < 0 {
			allNonNeg = false
		}
		sum += v
	}
	if allNonNeg && sum > 0.95 && sum < 1.05 {
		copy(probs, rawOutputs)
		return probs
	}
	maxVal := rawOutputs[0]
	for _, v := range rawOutputs {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := float32(0)
	for i, v := range rawOutputs {
		p := float32(math.Exp(float64(v - maxVal)))
		probs[i] = p
		expSum += p
	}
	if expSum > 0 {
		for i := range probs {
			probs[i] /= expSum
		}
	}
	return probs
}

func drawViewerButton(screen *ebiten.Image, font *text.GoTextFace, label string, x, y, w, h float32, isHover, isAccent bool) {
	bg := color.RGBA{R: 32, G: 35, B: 42, A: 255}
	border := LINE_COLOR
	textColor := color.RGBA{R: 220, G: 225, B: 235, A: 255}

	if isAccent {
		bg = color.RGBA{R: 32, G: 50, B: 75, A: 255}
		border = COLOR_1
		textColor = color.RGBA{R: 240, G: 245, B: 255, A: 255}
		if isHover {
			bg = color.RGBA{R: 45, G: 72, B: 110, A: 255}
			border = color.RGBA{R: 120, G: 200, B: 255, A: 255}
		}
	} else if isHover {
		bg = color.RGBA{R: 48, G: 52, B: 62, A: 255}
		border = COLOR_1
	}

	vector.FillRect(screen, x, y, w, h, bg, true)
	vector.StrokeRect(screen, x, y, w, h, 1, border, true)

	op := &text.DrawOptions{}
	textW := float32(len(label)) * 6.8
	textX := x + (w-textW)/2
	if textX < x+8 {
		textX = x + 8
	}
	textY := y + (h-14)/2
	op.GeoM.Translate(float64(textX), float64(textY))
	op.ColorScale.ScaleWithColor(textColor)
	text.Draw(screen, label, font, op)
}

func (g *Game) updateDigitViewerTab(mx, my float32) {
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowLeft) || inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.PrevViewerImage()
	} else if inpututil.IsKeyJustPressed(ebiten.KeyArrowRight) || inpututil.IsKeyJustPressed(ebiten.KeyN) {
		g.NextViewerImage()
	} else if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		g.RandomViewerImage()
	} else if inpututil.IsKeyJustPressed(ebiten.KeyF) || inpututil.IsKeyJustPressed(ebiten.KeyW) || inpututil.IsKeyJustPressed(ebiten.KeyM) {
		g.FindNextMisclassified()
	}

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		contentY := float32(TAB_BAR_HEIGHT + TOP_BAR_HEIGHT + 15)
		card1X := float32(20)
		card1Y := contentY

		btnRow1Y := card1Y + 390
		btnH1 := float32(34)
		btnPrevX := card1X + 16
		btnPrevW := float32(115)

		btnRandX := card1X + 140
		btnRandW := float32(120)

		btnNextX := card1X + 269
		btnNextW := float32(115)

		btnRow2Y := card1Y + 434
		btnFindX := card1X + 16
		btnFindW := float32(368)
		btnFindH := float32(38)

		if my >= btnRow1Y && my <= btnRow1Y+btnH1 {
			if mx >= btnPrevX && mx <= btnPrevX+btnPrevW {
				g.PrevViewerImage()
			} else if mx >= btnRandX && mx <= btnRandX+btnRandW {
				g.RandomViewerImage()
			} else if mx >= btnNextX && mx <= btnNextX+btnNextW {
				g.NextViewerImage()
			}
		} else if my >= btnRow2Y && my <= btnRow2Y+btnFindH {
			if mx >= btnFindX && mx <= btnFindX+btnFindW {
				g.FindNextMisclassified()
			}
		}
	}
}

func (g *Game) drawDigitViewerTab(screen *ebiten.Image) {
	contentY := float32(TAB_BAR_HEIGHT)
	contentW := float32(g.screenWidth)

	bannerH := float32(TOP_BAR_HEIGHT)
	vector.FillRect(screen, 0, contentY, contentW, bannerH, COLOR_4, true)
	vector.StrokeLine(screen, 0, contentY+bannerH, contentW, contentY+bannerH, 1, LINE_COLOR, true)

	headerOp := &text.DrawOptions{}
	headerOp.GeoM.Translate(15, float64(contentY+12))
	headerOp.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, "MNIST INSPECTOR & DIGIT CLASSIFIER", g.fontFace, headerOp)

	total := g.TotalViewerImages()
	infoOp := &text.DrawOptions{}
	infoOp.GeoM.Translate(320, float64(contentY+12))
	infoOp.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, fmt.Sprintf("Image %d of %d  |  Train Acc: %.1f%%  |  Test Acc: %.1f%%", g.viewerIndex+1, total, g.trainAccuracy, g.testAccuracy), g.fontFace, infoOp)

	if g.isLearning {
		badgeW := float32(230)
		badgeH := float32(24)
		badgeX := contentW - badgeW - 20
		badgeY := contentY + 8
		if badgeX > 560 {
			vector.FillRect(screen, badgeX, badgeY, badgeW, badgeH, color.RGBA{R: 20, G: 55, B: 30, A: 255}, true)
			vector.StrokeRect(screen, badgeX, badgeY, badgeW, badgeH, 1, color.RGBA{R: 76, G: 175, B: 80, A: 255}, true)
			bOp := &text.DrawOptions{}
			bOp.GeoM.Translate(float64(badgeX+12), float64(badgeY+5))
			bOp.ColorScale.ScaleWithColor(color.RGBA{R: 120, G: 240, B: 130, A: 255})
			text.Draw(screen, "● network is learning", g.fontFace, bOp)
		}
	}

	mainY := contentY + bannerH + 15
	mainH := float32(g.screenHeight) - mainY - 15

	card1X := float32(20)
	card1Y := mainY
	card1W := float32(400)
	card1H := mainH

	// LEFT CARD: Image & Controls
	vector.FillRect(screen, card1X, card1Y, card1W, card1H, COLOR_4, true)
	vector.StrokeRect(screen, card1X, card1Y, card1W, card1H, 1, LINE_COLOR, true)

	c1TitleOp := &text.DrawOptions{}
	c1TitleOp.GeoM.Translate(float64(card1X+16), float64(card1Y+14))
	c1TitleOp.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, "MNIST DIGIT IMAGE", g.fontFace, c1TitleOp)

	c1SubOp := &text.DrawOptions{}
	c1SubOp.GeoM.Translate(float64(card1X+16), float64(card1Y+34))
	c1SubOp.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, fmt.Sprintf("Sample #%d of %d (28x28 grayscale)", g.viewerIndex+1, total), g.fontFace, c1SubOp)

	// Digit Viewport: 280x280 (10x scaled)
	imgX := card1X + (card1W-280)/2
	imgY := card1Y + 56
	vector.FillRect(screen, imgX-2, imgY-2, 284, 284, color.RGBA{R: 12, G: 14, B: 18, A: 255}, true)
	vector.StrokeRect(screen, imgX-2, imgY-2, 284, 284, 1.5, COLOR_1, true)

	if g.viewerImage != nil {
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Scale(10.0, 10.0)
		op.GeoM.Translate(float64(imgX), float64(imgY))
		op.Filter = ebiten.FilterNearest
		screen.DrawImage(g.viewerImage, op)
	} else {
		noImgOp := &text.DrawOptions{}
		noImgOp.GeoM.Translate(float64(imgX+70), float64(imgY+130))
		noImgOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "No image loaded", g.fontFace, noImgOp)
	}

	pixels, trueLabel, _, _ := g.GetViewerSample(g.viewerIndex)

	// True Label Badge
	lblY := card1Y + 348
	lblW := card1W - 32
	lblX := card1X + 16
	vector.FillRect(screen, lblX, lblY, lblW, 32, color.RGBA{R: 24, G: 28, B: 36, A: 255}, true)
	vector.StrokeRect(screen, lblX, lblY, lblW, 32, 1, LINE_COLOR, true)

	trueLblOp := &text.DrawOptions{}
	trueLblOp.GeoM.Translate(float64(lblX+14), float64(lblY+8))
	trueLblOp.ColorScale.ScaleWithColor(color.RGBA{R: 200, G: 205, B: 215, A: 255})
	text.Draw(screen, fmt.Sprintf("Ground Truth Label:  Digit %d", trueLabel), g.fontFace, trueLblOp)

	mx, my := ebiten.CursorPosition()
	fmx, fmy := float32(mx), float32(my)

	btnRow1Y := card1Y + 390
	btnH1 := float32(34)
	btnPrevX := card1X + 16
	btnPrevW := float32(115)

	btnRandX := card1X + 140
	btnRandW := float32(120)

	btnNextX := card1X + 269
	btnNextW := float32(115)

	isPrevHover := fmx >= btnPrevX && fmx <= btnPrevX+btnPrevW && fmy >= btnRow1Y && fmy <= btnRow1Y+btnH1
	drawViewerButton(screen, g.fontFace, "◀  Prev", btnPrevX, btnRow1Y, btnPrevW, btnH1, isPrevHover, false)

	isRandHover := fmx >= btnRandX && fmx <= btnRandX+btnRandW && fmy >= btnRow1Y && fmy <= btnRow1Y+btnH1
	drawViewerButton(screen, g.fontFace, "🎲  Random", btnRandX, btnRow1Y, btnRandW, btnH1, isRandHover, false)

	isNextHover := fmx >= btnNextX && fmx <= btnNextX+btnNextW && fmy >= btnRow1Y && fmy <= btnRow1Y+btnH1
	drawViewerButton(screen, g.fontFace, "Next  ▶", btnNextX, btnRow1Y, btnNextW, btnH1, isNextHover, false)

	btnRow2Y := card1Y + 434
	btnFindX := card1X + 16
	btnFindW := card1W - 32
	btnFindH := float32(38)
	isFindHover := fmx >= btnFindX && fmx <= btnFindX+btnFindW && fmy >= btnRow2Y && fmy <= btnRow2Y+btnFindH
	drawViewerButton(screen, g.fontFace, "🔍  FIND NEXT MISCLASSIFIED DIGIT", btnFindX, btnRow2Y, btnFindW, btnFindH, isFindHover, true)

	statusY := card1Y + 482
	statusW := card1W - 32
	statusX := card1X + 16
	if g.viewerStatusMsg != "" {
		statBg := color.RGBA{R: 24, G: 28, B: 36, A: 255}
		statBorder := LINE_COLOR
		statTextColor := COLOR_3
		if strings.Contains(g.viewerStatusMsg, "misclassified") {
			statBg = color.RGBA{R: 50, G: 32, B: 20, A: 255}
			statBorder = color.RGBA{R: 255, G: 167, B: 38, A: 255}
			statTextColor = color.RGBA{R: 255, G: 215, B: 140, A: 255}
		} else if strings.Contains(g.viewerStatusMsg, "100%") {
			statBg = color.RGBA{R: 20, G: 50, B: 30, A: 255}
			statBorder = color.RGBA{R: 76, G: 215, B: 100, A: 255}
			statTextColor = color.RGBA{R: 130, G: 255, B: 160, A: 255}
		}
		vector.FillRect(screen, statusX, statusY, statusW, 34, statBg, true)
		vector.StrokeRect(screen, statusX, statusY, statusW, 34, 1, statBorder, true)
		stOp := &text.DrawOptions{}
		stOp.GeoM.Translate(float64(statusX+10), float64(statusY+9))
		stOp.ColorScale.ScaleWithColor(statTextColor)
		text.Draw(screen, g.viewerStatusMsg, g.fontFace, stOp)
	}

	hintOp := &text.DrawOptions{}
	hintOp.GeoM.Translate(float64(card1X+16), float64(card1Y+530))
	hintOp.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, "Keys: [← / →] Prev/Next  |  [R] Random  |  [F] Find Wrong", g.fontFace, hintOp)

	// RIGHT CARD: Prediction & Confidence
	card2X := card1X + card1W + 15
	card2Y := mainY
	card2W := contentW - card2X - 20
	card2H := mainH

	if card2W > 200 {
		vector.FillRect(screen, card2X, card2Y, card2W, card2H, COLOR_4, true)
		vector.StrokeRect(screen, card2X, card2Y, card2W, card2H, 1, LINE_COLOR, true)

		c2TitleOp := &text.DrawOptions{}
		c2TitleOp.GeoM.Translate(float64(card2X+20), float64(card2Y+14))
		c2TitleOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "NETWORK PREDICTION & CONFIDENCE", g.fontFace, c2TitleOp)

		c2SubOp := &text.DrawOptions{}
		c2SubOp.GeoM.Translate(float64(card2X+20), float64(card2Y+34))
		c2SubOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Forward pass through current active weights", g.fontFace, c2SubOp)

		var probs []float32
		predDigit := -1
		confidence := float32(0)
		inputs := g.getViewerNNInputs(pixels)
		if len(inputs) > 0 {
			rawOutputs := CalculateOutputs(g.nn, inputs)
			probs = getNormalizedProbabilities(rawOutputs)
			predDigit = IndexOfMaxValue(rawOutputs)
			if predDigit >= 0 && predDigit < len(probs) {
				confidence = probs[predDigit] * 100.0
			}
		}

		bannerY := card2Y + 56
		bannerW := card2W - 40
		bannerX := card2X + 20
		bannerH := float32(50)

		if predDigit >= 0 {
			isCorrect := predDigit == trueLabel
			vBg := color.RGBA{R: 20, G: 55, B: 30, A: 255}
			vBorder := color.RGBA{R: 76, G: 215, B: 100, A: 255}
			vTitle := fmt.Sprintf("✓  CORRECT PREDICTION:  DIGIT %d", predDigit)
			vSub := fmt.Sprintf("Confidence: %.1f%%  |  Matches ground truth label", confidence)
			if !isCorrect {
				vBg = color.RGBA{R: 65, G: 25, B: 25, A: 255}
				vBorder = COLOR_2
				vTitle = fmt.Sprintf("✗  MISCLASSIFIED:  PREDICTED %d  (TRUE %d)", predDigit, trueLabel)
				vSub = fmt.Sprintf("Confidence: %.1f%%  |  Network is mistaken on this digit", confidence)
			}
			vector.FillRect(screen, bannerX, bannerY, bannerW, bannerH, vBg, true)
			vector.StrokeRect(screen, bannerX, bannerY, bannerW, bannerH, 1.5, vBorder, true)

			vOp1 := &text.DrawOptions{}
			vOp1.GeoM.Translate(float64(bannerX+14), float64(bannerY+9))
			vOp1.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 255, B: 255, A: 255})
			text.Draw(screen, vTitle, g.fontFace, vOp1)

			vOp2 := &text.DrawOptions{}
			vOp2.GeoM.Translate(float64(bannerX+14), float64(bannerY+29))
			vOp2.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, vSub, g.fontFace, vOp2)
		} else {
			vector.FillRect(screen, bannerX, bannerY, bannerW, bannerH, color.RGBA{R: 25, G: 28, B: 35, A: 255}, true)
			vector.StrokeRect(screen, bannerX, bannerY, bannerW, bannerH, 1, LINE_COLOR, true)
			vOp1 := &text.DrawOptions{}
			vOp1.GeoM.Translate(float64(bannerX+14), float64(bannerY+16))
			vOp1.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, "No active prediction available", g.fontFace, vOp1)
		}

		probHeadOp := &text.DrawOptions{}
		probHeadOp.GeoM.Translate(float64(card2X+20), float64(card2Y+124))
		probHeadOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "Class Probabilities (Digits 0 – 9):", g.fontFace, probHeadOp)

		barsStartY := card2Y + 148
		rowH := float32(28)
		trackX := card2X + 85
		trackW := card2W - 170
		if trackW < 100 {
			trackW = 100
		}
		trackH := float32(16)

		for d := 0; d < 10; d++ {
			rowY := barsStartY + float32(d)*rowH

			lblText := fmt.Sprintf("Digit %d", d)
			dOp := &text.DrawOptions{}
			dOp.GeoM.Translate(float64(card2X+20), float64(rowY+2))
			if d == trueLabel {
				dOp.ColorScale.ScaleWithColor(color.RGBA{R: 100, G: 220, B: 255, A: 255})
			} else {
				dOp.ColorScale.ScaleWithColor(COLOR_3)
			}
			text.Draw(screen, lblText, g.fontFace, dOp)

			vector.FillRect(screen, trackX, rowY+3, trackW, trackH, color.RGBA{R: 22, G: 25, B: 30, A: 255}, true)
			vector.StrokeRect(screen, trackX, rowY+3, trackW, trackH, 1, LINE_COLOR, true)

			p := float32(0)
			if d < len(probs) {
				p = probs[d]
			}
			fillW := trackW * p
			if fillW > trackW {
				fillW = trackW
			}

			barColor := color.RGBA{R: 60, G: 80, B: 120, A: 255}
			if d == predDigit && d == trueLabel {
				barColor = color.RGBA{R: 76, G: 215, B: 100, A: 255}
			} else if d == predDigit && d != trueLabel {
				barColor = color.RGBA{R: 235, G: 87, B: 87, A: 255}
			} else if d == trueLabel {
				barColor = color.RGBA{R: 65, G: 160, B: 220, A: 255}
			}

			if fillW > 1 {
				vector.FillRect(screen, trackX+1, rowY+4, fillW-2, trackH-2, barColor, true)
			}

			pctOp := &text.DrawOptions{}
			pctOp.GeoM.Translate(float64(trackX+trackW+10), float64(rowY+2))
			if d == predDigit {
				pctOp.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 255, B: 255, A: 255})
			} else if d == trueLabel {
				pctOp.ColorScale.ScaleWithColor(color.RGBA{R: 100, G: 220, B: 255, A: 255})
			} else {
				pctOp.ColorScale.ScaleWithColor(COLOR_3)
			}
			marker := ""
			if d == trueLabel {
				marker = " (True)"
			}
			text.Draw(screen, fmt.Sprintf("%5.1f%%%s", p*100.0, marker), g.fontFace, pctOp)
		}

		sumBoxY := card2Y + 440
		sumBoxW := card2W - 40
		sumBoxX := card2X + 20
		sumBoxH := float32(80)
		if sumBoxY+sumBoxH <= card2Y+card2H-15 {
			vector.FillRect(screen, sumBoxX, sumBoxY, sumBoxW, sumBoxH, color.RGBA{R: 22, G: 25, B: 30, A: 255}, true)
			vector.StrokeRect(screen, sumBoxX, sumBoxY, sumBoxW, sumBoxH, 1, LINE_COLOR, true)

			s1Op := &text.DrawOptions{}
			s1Op.GeoM.Translate(float64(sumBoxX+14), float64(sumBoxY+10))
			s1Op.ColorScale.ScaleWithColor(COLOR_1)
			text.Draw(screen, fmt.Sprintf("Network Architecture:  %v", g.layerSizes), g.fontFace, s1Op)

			s2Op := &text.DrawOptions{}
			s2Op.GeoM.Translate(float64(sumBoxX+14), float64(sumBoxY+32))
			s2Op.ColorScale.ScaleWithColor(COLOR_3)
			actName := "LeakyReLU"
			outName := "Softmax"
			if g.dropdownActivation != nil {
				actName = g.dropdownActivation.SelectedValue()
			}
			if g.dropdownOutput != nil {
				outName = g.dropdownOutput.SelectedValue()
			}
			text.Draw(screen, fmt.Sprintf("Activation: %s  |  Output: %s  |  Learning Rate: %.4f", actName, outName, g.lr), g.fontFace, s2Op)

			s3Op := &text.DrawOptions{}
			s3Op.GeoM.Translate(float64(sumBoxX+14), float64(sumBoxY+54))
			s3Op.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, fmt.Sprintf("Dataset Performance: Train Acc %.1f%%  |  Test Acc %.1f%%  |  Epochs %d", g.trainAccuracy, g.testAccuracy, g.totalEpochs), g.fontFace, s3Op)
		}
	}
}

func (g *Game) updateCanvasTab(mx, my float32) {
	if inpututil.IsKeyJustPressed(ebiten.KeyC) {
		g.ClearCanvas()
	}

	contentY := float32(TAB_BAR_HEIGHT + TOP_BAR_HEIGHT + 15)
	card1X := float32(20)
	card1Y := contentY
	card1W := float32(400)

	// Canvas Viewport: 280x280
	canvasX := card1X + (card1W-280)/2
	canvasY := card1Y + 56
	canvasW := float32(280)
	canvasH := float32(280)

	// Clear button: [ 🗑 CLEAR CANVAS ]
	btnClearX := card1X + 16
	btnClearY := card1Y + 356
	btnClearW := card1W - 32
	btnClearH := float32(38)

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		if mx >= btnClearX && mx <= btnClearX+btnClearW && my >= btnClearY && my <= btnClearY+btnClearH {
			g.ClearCanvas()
			return
		}
	}

	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		if mx >= canvasX && mx < canvasX+canvasW && my >= canvasY && my < canvasY+canvasH {
			gx := (mx - canvasX) / 10.0
			gy := (my - canvasY) / 10.0
			if g.isDrawingActive {
				g.PaintCanvasStroke(g.drawPrevX, g.drawPrevY, gx, gy)
			} else {
				g.PaintCanvasPoint(gx, gy)
				g.updateCanvasImage()
			}
			g.drawPrevX = gx
			g.drawPrevY = gy
			g.isDrawingActive = true
		} else {
			g.isDrawingActive = false
		}
	} else {
		g.isDrawingActive = false
	}
}

func (g *Game) drawCanvasTab(screen *ebiten.Image) {
	contentY := float32(TAB_BAR_HEIGHT)
	contentW := float32(g.screenWidth)

	bannerH := float32(TOP_BAR_HEIGHT)
	vector.FillRect(screen, 0, contentY, contentW, bannerH, COLOR_4, true)
	vector.StrokeLine(screen, 0, contentY+bannerH, contentW, contentY+bannerH, 1, LINE_COLOR, true)

	headerOp := &text.DrawOptions{}
	headerOp.GeoM.Translate(15, float64(contentY+12))
	headerOp.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, "INTERACTIVE DOODLE CANVAS & LIVE PREDICTOR", g.fontFace, headerOp)

	infoOp := &text.DrawOptions{}
	infoOp.GeoM.Translate(360, float64(contentY+12))
	infoOp.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, fmt.Sprintf("Live Drawing  |  Train Acc: %.1f%%  |  Test Acc: %.1f%%", g.trainAccuracy, g.testAccuracy), g.fontFace, infoOp)

	if g.isLearning {
		badgeW := float32(230)
		badgeH := float32(24)
		badgeX := contentW - badgeW - 20
		badgeY := contentY + 8
		if badgeX > 620 {
			vector.FillRect(screen, badgeX, badgeY, badgeW, badgeH, color.RGBA{R: 20, G: 55, B: 30, A: 255}, true)
			vector.StrokeRect(screen, badgeX, badgeY, badgeW, badgeH, 1, color.RGBA{R: 76, G: 175, B: 80, A: 255}, true)
			bOp := &text.DrawOptions{}
			bOp.GeoM.Translate(float64(badgeX+12), float64(badgeY+5))
			bOp.ColorScale.ScaleWithColor(color.RGBA{R: 120, G: 240, B: 130, A: 255})
			text.Draw(screen, "● network is learning", g.fontFace, bOp)
		}
	}

	mainY := contentY + bannerH + 15
	mainH := float32(g.screenHeight) - mainY - 15

	card1X := float32(20)
	card1Y := mainY
	card1W := float32(400)
	card1H := mainH

	// LEFT CARD: Drawing Canvas & Controls
	vector.FillRect(screen, card1X, card1Y, card1W, card1H, COLOR_4, true)
	vector.StrokeRect(screen, card1X, card1Y, card1W, card1H, 1, LINE_COLOR, true)

	c1TitleOp := &text.DrawOptions{}
	c1TitleOp.GeoM.Translate(float64(card1X+16), float64(card1Y+14))
	c1TitleOp.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, "DOODLE CANVAS (DRAW HERE)", g.fontFace, c1TitleOp)

	c1SubOp := &text.DrawOptions{}
	c1SubOp.GeoM.Translate(float64(card1X+16), float64(card1Y+34))
	c1SubOp.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, "Draw any digit (0–9) using your mouse", g.fontFace, c1SubOp)

	// Canvas Viewport: 280x280
	canvasX := card1X + (card1W-280)/2
	canvasY := card1Y + 56
	vector.FillRect(screen, canvasX-2, canvasY-2, 284, 284, color.RGBA{R: 0, G: 0, B: 0, A: 255}, true)
	vector.StrokeRect(screen, canvasX-2, canvasY-2, 284, 284, 1.5, COLOR_1, true)

	if g.drawImage != nil {
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Scale(10.0, 10.0)
		op.GeoM.Translate(float64(canvasX), float64(canvasY))
		op.Filter = ebiten.FilterNearest
		screen.DrawImage(g.drawImage, op)
	}

	isEmpty := g.isCanvasEmpty()
	if isEmpty {
		hintCenterOp := &text.DrawOptions{}
		hintCenterOp.GeoM.Translate(float64(canvasX+45), float64(canvasY+132))
		hintCenterOp.ColorScale.ScaleWithColor(color.RGBA{R: 80, G: 85, B: 95, A: 255})
		text.Draw(screen, "Click & drag here to draw...", g.fontFace, hintCenterOp)
	}

	mx, my := ebiten.CursorPosition()
	fmx, fmy := float32(mx), float32(my)

	// Clear Button: [ 🗑 CLEAR CANVAS ]
	btnClearX := card1X + 16
	btnClearY := card1Y + 356
	btnClearW := card1W - 32
	btnClearH := float32(38)
	isClearHover := fmx >= btnClearX && fmx <= btnClearX+btnClearW && fmy >= btnClearY && fmy <= btnClearY+btnClearH
	drawViewerButton(screen, g.fontFace, "🗑  CLEAR CANVAS", btnClearX, btnClearY, btnClearW, btnClearH, isClearHover, true)

	// Drawing hints
	tipOp1 := &text.DrawOptions{}
	tipOp1.GeoM.Translate(float64(card1X+16), float64(card1Y+415))
	tipOp1.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, "Tips: Left-click and drag across the black box to draw.", g.fontFace, tipOp1)

	tipOp2 := &text.DrawOptions{}
	tipOp2.GeoM.Translate(float64(card1X+16), float64(card1Y+438))
	tipOp2.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, "Hotkeys: [C] Clear  |  [1-4] Switch tabs", g.fontFace, tipOp2)

	// RIGHT CARD: Real-time Prediction & Confidence
	card2X := card1X + card1W + 15
	card2Y := mainY
	card2W := contentW - card2X - 20
	card2H := mainH

	if card2W > 200 {
		vector.FillRect(screen, card2X, card2Y, card2W, card2H, COLOR_4, true)
		vector.StrokeRect(screen, card2X, card2Y, card2W, card2H, 1, LINE_COLOR, true)

		c2TitleOp := &text.DrawOptions{}
		c2TitleOp.GeoM.Translate(float64(card2X+20), float64(card2Y+14))
		c2TitleOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "REAL-TIME NETWORK PREDICTION", g.fontFace, c2TitleOp)

		c2SubOp := &text.DrawOptions{}
		c2SubOp.GeoM.Translate(float64(card2X+20), float64(card2Y+34))
		c2SubOp.ColorScale.ScaleWithColor(COLOR_3)
		text.Draw(screen, "Live forward pass inference on user drawing", g.fontFace, c2SubOp)

		var probs []float32
		predDigit := -1
		confidence := float32(0)

		if !isEmpty {
			inputs := g.getViewerNNInputs(g.drawPixels)
			if len(inputs) > 0 {
				rawOutputs := CalculateOutputs(g.nn, inputs)
				probs = getNormalizedProbabilities(rawOutputs)
				predDigit = IndexOfMaxValue(rawOutputs)
				if predDigit >= 0 && predDigit < len(probs) {
					confidence = probs[predDigit] * 100.0
				}
			}
		}

		bannerY := card2Y + 56
		bannerW := card2W - 40
		bannerX := card2X + 20
		bannerH := float32(50)

		if !isEmpty && predDigit >= 0 {
			vBg := color.RGBA{R: 20, G: 55, B: 45, A: 255}
			vBorder := color.RGBA{R: 76, G: 215, B: 150, A: 255}
			vTitle := fmt.Sprintf("●  LIVE RECOGNITION:  DIGIT %d", predDigit)
			vSub := fmt.Sprintf("Confidence: %.1f%%  |  Network recognizes digit %d", confidence, predDigit)

			vector.FillRect(screen, bannerX, bannerY, bannerW, bannerH, vBg, true)
			vector.StrokeRect(screen, bannerX, bannerY, bannerW, bannerH, 1.5, vBorder, true)

			vOp1 := &text.DrawOptions{}
			vOp1.GeoM.Translate(float64(bannerX+14), float64(bannerY+9))
			vOp1.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 255, B: 255, A: 255})
			text.Draw(screen, vTitle, g.fontFace, vOp1)

			vOp2 := &text.DrawOptions{}
			vOp2.GeoM.Translate(float64(bannerX+14), float64(bannerY+29))
			vOp2.ColorScale.ScaleWithColor(color.RGBA{R: 180, G: 240, B: 210, A: 255})
			text.Draw(screen, vSub, g.fontFace, vOp2)
		} else {
			vector.FillRect(screen, bannerX, bannerY, bannerW, bannerH, color.RGBA{R: 25, G: 28, B: 35, A: 255}, true)
			vector.StrokeRect(screen, bannerX, bannerY, bannerW, bannerH, 1, LINE_COLOR, true)
			vOp1 := &text.DrawOptions{}
			vOp1.GeoM.Translate(float64(bannerX+14), float64(bannerY+9))
			vOp1.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, "[ DRAW A DIGIT ON THE CANVAS ]", g.fontFace, vOp1)

			vOp2 := &text.DrawOptions{}
			vOp2.GeoM.Translate(float64(bannerX+14), float64(bannerY+29))
			vOp2.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, "The neural network will analyze your drawing in real time", g.fontFace, vOp2)
		}

		probHeadOp := &text.DrawOptions{}
		probHeadOp.GeoM.Translate(float64(card2X+20), float64(card2Y+124))
		probHeadOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "Class Probabilities (Digits 0 – 9):", g.fontFace, probHeadOp)

		barsStartY := card2Y + 148
		rowH := float32(28)
		trackX := card2X + 85
		trackW := card2W - 170
		if trackW < 100 {
			trackW = 100
		}
		trackH := float32(16)

		for d := 0; d < 10; d++ {
			rowY := barsStartY + float32(d)*rowH

			lblText := fmt.Sprintf("Digit %d", d)
			dOp := &text.DrawOptions{}
			dOp.GeoM.Translate(float64(card2X+20), float64(rowY+2))
			if d == predDigit && !isEmpty {
				dOp.ColorScale.ScaleWithColor(color.RGBA{R: 100, G: 240, B: 200, A: 255})
			} else {
				dOp.ColorScale.ScaleWithColor(COLOR_3)
			}
			text.Draw(screen, lblText, g.fontFace, dOp)

			vector.FillRect(screen, trackX, rowY+3, trackW, trackH, color.RGBA{R: 22, G: 25, B: 30, A: 255}, true)
			vector.StrokeRect(screen, trackX, rowY+3, trackW, trackH, 1, LINE_COLOR, true)

			p := float32(0)
			if d < len(probs) && !isEmpty {
				p = probs[d]
			}
			fillW := trackW * p
			if fillW > trackW {
				fillW = trackW
			}

			barColor := color.RGBA{R: 60, G: 80, B: 120, A: 255}
			if d == predDigit && !isEmpty {
				barColor = color.RGBA{R: 76, G: 215, B: 140, A: 255}
			}

			if fillW > 1 {
				vector.FillRect(screen, trackX+1, rowY+4, fillW-2, trackH-2, barColor, true)
			}

			pctOp := &text.DrawOptions{}
			pctOp.GeoM.Translate(float64(trackX+trackW+10), float64(rowY+2))
			if d == predDigit && !isEmpty {
				pctOp.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 255, B: 255, A: 255})
			} else {
				pctOp.ColorScale.ScaleWithColor(COLOR_3)
			}
			marker := ""
			if d == predDigit && !isEmpty {
				marker = " (Top)"
			}
			text.Draw(screen, fmt.Sprintf("%5.1f%%%s", p*100.0, marker), g.fontFace, pctOp)
		}

		sumBoxY := card2Y + 440
		sumBoxW := card2W - 40
		sumBoxX := card2X + 20
		sumBoxH := float32(80)
		if sumBoxY+sumBoxH <= card2Y+card2H-15 {
			vector.FillRect(screen, sumBoxX, sumBoxY, sumBoxW, sumBoxH, color.RGBA{R: 22, G: 25, B: 30, A: 255}, true)
			vector.StrokeRect(screen, sumBoxX, sumBoxY, sumBoxW, sumBoxH, 1, LINE_COLOR, true)

			s1Op := &text.DrawOptions{}
			s1Op.GeoM.Translate(float64(sumBoxX+14), float64(sumBoxY+10))
			s1Op.ColorScale.ScaleWithColor(COLOR_1)
			text.Draw(screen, fmt.Sprintf("Network Architecture:  %v", g.layerSizes), g.fontFace, s1Op)

			s2Op := &text.DrawOptions{}
			s2Op.GeoM.Translate(float64(sumBoxX+14), float64(sumBoxY+32))
			s2Op.ColorScale.ScaleWithColor(COLOR_3)
			actName := "LeakyReLU"
			outName := "Softmax"
			if g.dropdownActivation != nil {
				actName = g.dropdownActivation.SelectedValue()
			}
			if g.dropdownOutput != nil {
				outName = g.dropdownOutput.SelectedValue()
			}
			text.Draw(screen, fmt.Sprintf("Activation: %s  |  Output: %s  |  Learning Rate: %.4f", actName, outName, g.lr), g.fontFace, s2Op)

			s3Op := &text.DrawOptions{}
			s3Op.GeoM.Translate(float64(sumBoxX+14), float64(sumBoxY+54))
			s3Op.ColorScale.ScaleWithColor(COLOR_3)
			text.Draw(screen, fmt.Sprintf("Dataset Performance: Train Acc %.1f%%  |  Test Acc %.1f%%  |  Epochs %d", g.trainAccuracy, g.testAccuracy, g.totalEpochs), g.fontFace, s3Op)
		}
	}
}

func DrawHUD(g *Game, screen *ebiten.Image, width float32) {
	hudY := TAB_BAR_HEIGHT
	vector.FillRect(screen, 0, hudY, width, TOP_BAR_HEIGHT, COLOR_4, true)
	vector.StrokeLine(screen, 0, hudY+TOP_BAR_HEIGHT, width, hudY+TOP_BAR_HEIGHT, 1, LINE_COLOR, true)

	mx, my := ebiten.CursorPosition()
	fmx, fmy := float32(mx), float32(my)
	btnY := hudY + 7
	btnH := float32(26)

	// 1. LEARN Button
	btnLearnX := float32(12)
	btnLearnW := float32(75)
	isLearnHover := fmx >= btnLearnX && fmx <= btnLearnX+btnLearnW && fmy >= btnY && fmy <= btnY+btnH

	if g.isLearning {
		learnBg := color.RGBA{R: 35, G: 125, B: 50, A: 255}
		if isLearnHover {
			learnBg = color.RGBA{R: 45, G: 155, B: 65, A: 255}
		}
		vector.FillRect(screen, btnLearnX, btnY, btnLearnW, btnH, learnBg, true)
		vector.StrokeRect(screen, btnLearnX, btnY, btnLearnW, btnH, 1, color.RGBA{R: 90, G: 200, B: 100, A: 255}, true)
		learnOp := &text.DrawOptions{}
		learnOp.GeoM.Translate(float64(btnLearnX+9), float64(btnY+6))
		learnOp.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 255, B: 255, A: 255})
		text.Draw(screen, "LEARNING", g.fontFace, learnOp)
	} else {
		learnBg := color.RGBA{R: 32, G: 34, B: 38, A: 255}
		if isLearnHover {
			learnBg = color.RGBA{R: 48, G: 52, B: 60, A: 255}
		}
		vector.FillRect(screen, btnLearnX, btnY, btnLearnW, btnH, learnBg, true)
		vector.StrokeRect(screen, btnLearnX, btnY, btnLearnW, btnH, 1, LINE_COLOR, true)
		learnOp := &text.DrawOptions{}
		learnOp.GeoM.Translate(float64(btnLearnX+18), float64(btnY+6))
		learnOp.ColorScale.ScaleWithColor(COLOR_1)
		text.Draw(screen, "LEARN", g.fontFace, learnOp)
	}

	// 2. KILL Switch Button
	btnKillX := float32(95)
	btnKillW := float32(60)
	isKillHover := fmx >= btnKillX && fmx <= btnKillX+btnKillW && fmy >= btnY && fmy <= btnY+btnH

	killBg := color.RGBA{R: 140, G: 30, B: 30, A: 255}
	if isKillHover {
		killBg = color.RGBA{R: 180, G: 40, B: 40, A: 255}
	}
	vector.FillRect(screen, btnKillX, btnY, btnKillW, btnH, killBg, true)
	vector.StrokeRect(screen, btnKillX, btnY, btnKillW, btnH, 1, color.RGBA{R: 230, G: 70, B: 70, A: 255}, true)
	killOp := &text.DrawOptions{}
	killOp.GeoM.Translate(float64(btnKillX+17), float64(btnY+6))
	killOp.ColorScale.ScaleWithColor(color.RGBA{R: 255, G: 240, B: 240, A: 255})
	text.Draw(screen, "KILL", g.fontFace, killOp)

	// Status text
	statusText := "[IDLE]"
	statusColor := COLOR_3
	if g.isKilled {
		statusText = "[KILLED]"
		statusColor = COLOR_2
	} else if g.isLearning {
		if g.isPaused {
			statusText = "[PAUSED]"
			statusColor = color.RGBA{R: 255, G: 167, B: 38, A: 255}
		} else {
			statusText = "[LEARNING]"
			statusColor = color.RGBA{R: 76, G: 175, B: 80, A: 255}
		}
	}

	op := &text.DrawOptions{}
	op.GeoM.Translate(168, float64(hudY+12))
	op.ColorScale.ScaleWithColor(statusColor)
	text.Draw(screen, fmt.Sprintf("STATUS: %s", statusText), g.fontFace, op)

	op.GeoM.Reset()
	op.GeoM.Translate(268, float64(hudY+12))
	op.ColorScale.Reset()
	op.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, fmt.Sprintf("COST: %.4f", g.cost), g.fontFace, op)

	op.GeoM.Reset()
	op.GeoM.Translate(365, float64(hudY+12))
	op.ColorScale.Reset()
	op.ColorScale.ScaleWithColor(COLOR_2)
	text.Draw(screen, fmt.Sprintf("ACC: %.1f%%", g.accuracy), g.fontFace, op)

	op.GeoM.Reset()
	op.GeoM.Translate(455, float64(hudY+12))
	op.ColorScale.Reset()
	op.ColorScale.ScaleWithColor(COLOR_2)
	text.Draw(screen, fmt.Sprintf("EPOCHS: %d", g.totalEpochs), g.fontFace, op)

	op.GeoM.Reset()
	op.GeoM.Translate(565, float64(hudY+12))
	op.ColorScale.Reset()
	op.ColorScale.ScaleWithColor(COLOR_3)
	text.Draw(screen, fmt.Sprintf("LR: %.5f", g.lr), g.fontFace, op)
}

func DrawSliders(g *Game, screen *ebiten.Image) {
	panelX := float32(g.screenWidth) - SLIDER_PANEL_WIDTH
	panelY := TAB_BAR_HEIGHT
	panelH := float32(g.screenHeight) - TAB_BAR_HEIGHT
	buttonWidth := float32(10)

	vector.FillRect(screen, panelX, panelY, SLIDER_PANEL_WIDTH, panelH, COLOR_4, true)
	vector.StrokeLine(screen, panelX, panelY, panelX, float32(g.screenHeight), 2, LINE_COLOR, true)

	for i, s := range g.paramDials {
		trackX, trackY, trackW, trackH := GetSliderLayout(g.screenWidth, g.screenHeight, i, g.sliderScrollY)

		// Viewport culling: skip drawing off-screen sliders
		if trackY < panelY+TOP_BAR_HEIGHT-10 || trackY > float32(g.screenHeight)+20 {
			continue
		}

		var val float32
		if s.name == "Weight" {
			val = g.nn.layers[s.l].weights[s.wi][s.wj]
		} else {
			val = g.nn.layers[s.l].biases[s.b]
		}

		op := &text.DrawOptions{}
		op.GeoM.Translate(float64(trackX), float64(trackY-18))
		op.ColorScale.ScaleWithColor(COLOR_2)
		displayText := fmt.Sprintf("%s %d: %.2f", s.name, i+1, val)
		text.Draw(screen, displayText, g.fontFace, op)

		vector.FillRect(screen, trackX, trackY+trackH/2-2, trackW, trackH/2, LINE_COLOR, true)

		buttonX := trackX + s.normValue*(trackW-buttonWidth)
		vector.FillRect(screen, buttonX, trackY, buttonWidth, trackH, COLOR_1, true)
	}

	// Header (drawn after sliders so scrolling sliders tuck behind the header)
	vector.FillRect(screen, panelX, panelY, SLIDER_PANEL_WIDTH, TOP_BAR_HEIGHT, COLOR_4, true)
	vector.StrokeLine(screen, panelX, panelY+TOP_BAR_HEIGHT, float32(g.screenWidth), panelY+TOP_BAR_HEIGHT, 1, LINE_COLOR, true)
	headerOp := &text.DrawOptions{}
	headerOp.GeoM.Translate(float64(panelX+15), float64(panelY+12))
	headerOp.ColorScale.ScaleWithColor(COLOR_1)
	text.Draw(screen, fmt.Sprintf("PARAMETERS (%d)", len(g.paramDials)), g.fontFace, headerOp)

	// Scroll indicator bar
	totalHeight := float32(len(g.paramDials))*SLIDER_SPACING + 70.0
	maxScroll := totalHeight - (panelH - TOP_BAR_HEIGHT)
	if maxScroll > 0 {
		scrollBarH := ((panelH - TOP_BAR_HEIGHT) / totalHeight) * (panelH - TOP_BAR_HEIGHT)
		if scrollBarH < 20 {
			scrollBarH = 20
		}
		scrollRatio := -g.sliderScrollY / maxScroll
		scrollBarY := panelY + TOP_BAR_HEIGHT + scrollRatio*(panelH-TOP_BAR_HEIGHT-scrollBarH)
		vector.FillRect(screen, float32(g.screenWidth)-5, scrollBarY, 3, scrollBarH, COLOR_1, true)
	}
}

func PlotPoints(g *Game, screen *ebiten.Image, plotX, plotY, plotW, plotH float32) {
	for _, point := range g.points {
		x := plotX
		if len(point.inputs) > 0 {
			x = plotX + point.inputs[0]*plotW
		}
		y := plotY + plotH/2
		if len(point.inputs) > 1 {
			y = (plotY + plotH) - point.inputs[1]*plotH
		}
		r := float32(5.0)
		classIdx := IndexOfMaxValue(point.outputs)
		c := GetClassColor(classIdx)
		vector.FillCircle(screen, x, y, r, c, true)
	}
}

func DrawGrid(screen *ebiten.Image, plotX, plotY, plotW, plotH float32) {
	for k := 0; k <= 10; k++ {
		ratio := float32(k) / 10.0
		gx := plotX + ratio*plotW
		vector.StrokeLine(screen, gx, plotY, gx, plotY+plotH, 1, LINE_COLOR, true)
		gy := (plotY + plotH) - ratio*plotH
		vector.StrokeLine(screen, plotX, gy, plotX+plotW, gy, 1, LINE_COLOR, true)
	}
	vector.StrokeLine(screen, plotX, plotY+plotH, plotX+plotW, plotY+plotH, 2, LINE_COLOR, true)
	vector.StrokeLine(screen, plotX, plotY, plotX, plotY+plotH, 2, LINE_COLOR, true)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	if outsideWidth < 500 {
		outsideWidth = 500
	}
	if outsideHeight < 400 {
		outsideHeight = 400
	}
	if g.screenWidth != outsideWidth || g.screenHeight != outsideHeight {
		g.screenWidth = outsideWidth
		g.screenHeight = outsideHeight
	}
	return outsideWidth, outsideHeight
}

func main() {
	// data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	// points := FruitToDataPoint(data)
	// data := ReadIrisDataset("./archive/iris/Iris.csv")
	// points := IrisToDataPoint(data)
	data := ReadMNISTDataset("./archive/mnist/train-images.idx3-ubyte", "./archive/mnist/train-labels.idx1-ubyte", 1000)
	points := MNISTToDataPoint(data)
	allPoints := make([]DataPoint, len(points))
	copy(allPoints, points)
	numInputs := len(points[0].inputs)
	numOutputs := len(points[0].outputs)
	layerSizes := []int{numInputs, 16, 32, 16, numOutputs}
	nn := NewNeuralNetwork(layerSizes)
	fontSource, err := text.NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		log.Fatalf("failed to parse font: %v", err)
	}
	fontFace := &text.GoTextFace{
		Source: fontSource,
		Size:   12,
	}

	boundaryImg := ebiten.NewImage(BOUNDARY_RESOLUTION, BOUNDARY_RESOLUTION)
	boundaryBuffer := make([]byte, BOUNDARY_RESOLUTION*BOUNDARY_RESOLUTION*4)

	ebiten.SetWindowSize(DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
	ebiten.SetWindowTitle("MNIST Neural Network Classifier")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	numFeatures := len(points[0].inputs)
	featureMeans := make([]float32, numFeatures)
	for _, pt := range points {
		for f := 0; f < numFeatures; f++ {
			featureMeans[f] += pt.inputs[f]
		}
	}
	for f := 0; f < numFeatures; f++ {
		featureMeans[f] /= float32(len(points))
	}
	boundaryInput := make([]float32, numFeatures)
	copy(boundaryInput, featureMeans)

	boundaryRenderer := NewBoundaryRenderer(nn, numFeatures, BOUNDARY_RESOLUTION)
	boundaryRenderer.SetFeatureMeans(featureMeans)

	paramDials := ParseNN(nn)
	trainPts, testPts := SplitDataset(allPoints, 80)
	initialTrainAcc := CalculateAccuracy(nn, trainPts)
	initialTestAcc := CalculateAccuracy(nn, testPts)
	game := &Game{
		screenWidth:         DEFAULT_WINDOW_W,
		screenHeight:        DEFAULT_WINDOW_H,
		currentTab:          TabDecisionBoundary,
		pixelImage:          boundaryImg,
		pixelBuffer:         boundaryBuffer,
		points:              trainPts,
		allPoints:           allPoints,
		trainPoints:         trainPts,
		testPoints:          testPts,
		batchSize:           32,
		trainSplitPercent:   80,
		inputLayerSizes:     fmt.Sprintf("%d, 16, 32, 16", numInputs),
		inputLearnRate:      "0.01",
		inputBatchSize:      "32",
		inputMomentum:       "0.9",
		dropdownActivation:  NewDropdown("Hidden Activation", []string{"LeakyReLU", "ReLU", "Sigmoid", "Tanh"}, 0),
		dropdownOutput:      NewDropdown("Output Function", []string{"Softmax", "Sigmoid", "Linear"}, 0),
		dropdownCost:        NewDropdown("Cost Function", []string{"Cross-Entropy", "MSE"}, 0),
		activeDropdownId:    0,
		activeInputId:       0,
		dialogSplitDragging: false,
		configError:         "",
		nn:                  nn,
		layerSizes:          layerSizes,
		paramDials:          paramDials,
		activeSlider:        -1,
		fontFace:            fontFace,
		cost:                NetworkCost(nn, trainPts),
		accuracy:            initialTestAcc,
		trainAccuracy:       initialTrainAcc,
		testAccuracy:        initialTestAcc,
		lr:                  0.01,
		tickCount:           0,
		isPaused:            false,
		isLearning:          false, // Starts in non-learning state!
		isKilled:            false,
		totalEpochs:         0,
		history:             []HistoryPoint{{Epoch: 0, TrainAccuracy: initialTrainAcc, TestAccuracy: initialTestAcc}},
		featureMeans:        featureMeans,
		boundaryInput:       boundaryInput,
		boundaryRenderer:    boundaryRenderer,
		mnistDataset:        data,
		viewerIndex:         0,
		drawPixels:          make([]float32, 784),
		drawImage:           ebiten.NewImage(28, 28),
	}

	PlotDecisionBoundary(game)
	game.SetViewerImage(0)
	game.ClearCanvas()
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
