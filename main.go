package main

import (
	"bytes"
	"fmt"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font/gofont/goregular"
)

type Point struct {
	x    uint
	y    uint
	flag uint
}

type Game struct {
	screenWidth  int
	screenHeight int
	pixelImage   *ebiten.Image
	pixelBuffer  []byte
	points       *[]DataPoint
	nn           *NeuralNetwork
	paramDials   []Slider
	activeSlider int
	fontFace     *text.GoTextFace
	cost         float32
	lr           float32
}

func (g *Game) Update() error {
	x, y := ebiten.CursorPosition()
	mx, my := float32(x), float32(y)
	buttonWidth := float32(18)

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		for i := range g.paramDials {
			trackX, trackY, trackW, trackH := GetSliderLayout(g.screenWidth, g.screenHeight, i)
			if mx >= trackX-5 && mx <= trackX+trackW+5 && my >= trackY-8 && my <= trackY+trackH+8 {
				g.activeSlider = i
				s := &g.paramDials[i]
				ratio := (mx - trackX - buttonWidth/2) / (trackW - buttonWidth)
				g.paramDials[i].normValue = ratio
				val := g.paramDials[i].Value()
				if s.name == "Weight" {
					g.nn.layers[s.l].weights[s.wi][s.wj] = val
				} else {
					g.nn.layers[s.l].biases[s.b] = val
				}
				PlotDecisionBoundary(g)
				break
			}
		}
	}

	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) && g.activeSlider != -1 {
		trackX, _, trackW, _ := GetSliderLayout(g.screenWidth, g.screenHeight, g.activeSlider)
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
		}
	}

	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		g.activeSlider = -1
	}
	println(g.lr)
	for j := 0; j < 60; j += 1 {
		for i := 0; i < 1000; i += 64 {
			numStart := i
			numEnd := i + 32
			if numEnd > 1000 {
				numEnd = 1000
			}
			Learn(g.nn, (*g.points)[numStart:numEnd], g.lr)
		}
	}
	g.lr -= 0.00000001
	UpdateSliders(g)
	PlotDecisionBoundary(g)
	cost := NetworkCost(g.nn, *g.points)
	g.cost = cost
	return nil
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
	for row := 0; row < BOUNDARY_RESOLUTION; row++ {
		dataY := 1.0 - float32(row)/float32(BOUNDARY_RESOLUTION)
		for col := 0; col < BOUNDARY_RESOLUTION; col++ {
			dataX := float32(col) / float32(BOUNDARY_RESOLUTION)
			ip := make([]float32, len((*g.points)[0].inputs))
			ip[0] = dataX
			ip[1] = dataY
			res := Classify(g.nn, ip)
			idx := (row*BOUNDARY_RESOLUTION + col) * 4

			var srcR, srcG, srcB float64
			var srcA float64

			if res == 0 {
				srcR, srcG, srcB = 66, 156, 245
				srcA = 100.0 / 255.0
			} else {
				srcR, srcG, srcB = 247, 74, 167
				srcA = 100.0 / 255.0
			}
			g.pixelBuffer[idx] = byte(srcR*srcA + (1.0 - srcA))
			g.pixelBuffer[idx+1] = byte(srcG*srcA + (1.0 - srcA))
			g.pixelBuffer[idx+2] = byte(srcB*srcA + (1.0 - srcA))
			g.pixelBuffer[idx+3] = 255
		}
	}
	g.pixelImage.WritePixels(g.pixelBuffer)
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(BG_COLOR)

	graphW := float32(g.screenWidth) - SLIDER_PANEL_WIDTH
	graphH := float32(g.screenHeight)
	marginX := float32(5)
	marginY := float32(5)
	plotW := graphW - marginX
	plotH := graphH - marginY

	if plotW > 0 && plotH > 0 {
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Scale(float64(plotW)/float64(BOUNDARY_RESOLUTION), float64(plotH)/float64(BOUNDARY_RESOLUTION))
		op.GeoM.Translate(float64(marginX), float64(graphH-marginY-plotH))
		screen.DrawImage(g.pixelImage, op)
		DrawGrid(screen, marginX, marginY, plotW, plotH, graphH)
		PlotPoints(g, screen, marginX, marginY, plotW, plotH, graphH)
	}

	DrawSliders(g, screen)
	DrawScore(g, screen)
}

func DrawScore(g *Game, screen *ebiten.Image) {
	panelX := float32(g.screenWidth) - SLIDER_PANEL_WIDTH - SLIDER_PANEL_WIDTH
	panelH := float32(50)
	vector.FillRect(screen, panelX, 0, SLIDER_PANEL_WIDTH, panelH, COLOR_4, true)
	vector.StrokeLine(screen, panelX, 0, panelX, panelH, 2, LINE_COLOR, true)
	op := &text.DrawOptions{}
	op.GeoM.Translate(float64(panelX+14), float64(panelH-40))
	op.ColorScale.ScaleWithColor(COLOR_2)
	displayText := fmt.Sprintf("COST: %.2f", g.cost)
	text.Draw(screen, displayText, g.fontFace, op)

}

func DrawSliders(g *Game, screen *ebiten.Image) {
	panelX := float32(g.screenWidth) - SLIDER_PANEL_WIDTH
	panelH := float32(g.screenHeight)
	buttonWidth := float32(10)

	vector.FillRect(screen, panelX, 0, SLIDER_PANEL_WIDTH, panelH, COLOR_4, true)
	vector.StrokeLine(screen, panelX, 0, panelX, panelH, 2, LINE_COLOR, true)

	for i, s := range g.paramDials {
		trackX, trackY, trackW, trackH := GetSliderLayout(g.screenWidth, g.screenHeight, i)

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
}

func PlotPoints(g *Game, screen *ebiten.Image, marginX, marginY, plotW, plotH, graphH float32) {
	for _, point := range *g.points {
		x := marginX + point.inputs[0]*plotW
		y := (graphH - marginY) - point.inputs[1]*plotH
		r := float32(5.0)
		if point.outputs[0] == 1 {
			vector.FillCircle(screen, x, y, r, COLOR_1, true)
		} else {
			vector.FillCircle(screen, x, y, r, COLOR_2, true)
		}
	}
}

func DrawGrid(screen *ebiten.Image, marginX, marginY, plotW, plotH, graphH float32) {
	for k := 0; k <= 10; k++ {
		gx := marginX + (float32(k)/10.0)*plotW
		vector.StrokeLine(screen, gx, graphH-marginY-plotH, gx, graphH-marginY, 1, LINE_COLOR, true)
		gy := (graphH - marginY) - (float32(k)/10.0)*plotH
		vector.StrokeLine(screen, marginX, gy, marginX+plotW, gy, 1, LINE_COLOR, true)
	}
	vector.StrokeLine(screen, marginX, graphH-marginY-10, marginX+plotW, graphH-marginY-10, 4, LINE_COLOR, true)
	vector.StrokeLine(screen, marginX+10, graphH-marginY-plotH, marginX+10, graphH-marginY, 4, LINE_COLOR, true)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	if g.screenWidth != outsideWidth || g.screenHeight != outsideHeight {
		g.screenWidth = outsideWidth
		g.screenHeight = outsideHeight
	}
	return outsideWidth, outsideHeight
}

func main() {
	data := ReadFruitDataset("./fruit_toxicity_dataset.csv")
	// data := ReadIrisDataset("./archive/Iris.csv")
	points := FruitToDataPoint(data)
	// points := IrisToDataPoint(data)
	nn := NewNeuralNetwork([]int{2, 4, 8, 8, 8, 8, 4, 2})
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
	ebiten.SetWindowTitle("Iris Neural Network Classifier")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	paramDials := ParseNN(nn)
	game := &Game{
		screenWidth:  DEFAULT_WINDOW_W,
		screenHeight: DEFAULT_WINDOW_H,
		pixelImage:   boundaryImg,
		pixelBuffer:  boundaryBuffer,
		points:       &points,
		nn:           nn,
		paramDials:   paramDials,
		activeSlider: -1,
		fontFace:     fontFace,
		cost:         0.0,
		lr:           0.01,
	}

	PlotDecisionBoundary(game)
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
