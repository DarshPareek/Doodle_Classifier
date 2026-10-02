package main

import (
	"fmt"
	"strconv"
	"strings"
)

type Slider struct {
	normValue float32 // 0.0 to 1.0 (0.5 represents 0.0 in range [-2.5, 2.5])
	name      string  // "Weight" or "Bias"
	l         int     // layer index
	wi        int     // weight row index
	wj        int     // weight col index
	b         int     // bias index
}

func NewSlider(name string, layer, wi, wj, b int) *Slider {
	return &Slider{
		normValue: 0.5,
		name:      name,
		l:         layer,
		wi:        wi,
		wj:        wj,
		b:         b,
	}
}

func (s *Slider) Value() float32 {
	return (s.normValue * float32(WEIGHT_RANGE_MULTIPLIER)) - float32(WEIGHT_RANGE_ADDER)
}

func (s *Slider) SetValue(val float32) {
	ratio := (val + float32(WEIGHT_RANGE_ADDER)) / float32(WEIGHT_RANGE_MULTIPLIER)
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	s.normValue = ratio
}

type TabType int

const (
	TabDecisionBoundary TabType = 0
	TabNewFeature       TabType = 1
	TabDigitViewer      TabType = 2
	TabCanvasDraw       TabType = 3
)

type Dropdown struct {
	Label         string
	Options       []string
	SelectedIndex int
	IsOpen        bool
}

func NewDropdown(label string, options []string, defaultIndex int) *Dropdown {
	if defaultIndex < 0 || defaultIndex >= len(options) {
		defaultIndex = 0
	}
	return &Dropdown{
		Label:         label,
		Options:       options,
		SelectedIndex: defaultIndex,
		IsOpen:        false,
	}
}

func (d *Dropdown) SelectedValue() string {
	if d.SelectedIndex >= 0 && d.SelectedIndex < len(d.Options) {
		return d.Options[d.SelectedIndex]
	}
	return ""
}

func (d *Dropdown) SelectIndex(idx int) {
	if idx >= 0 && idx < len(d.Options) {
		d.SelectedIndex = idx
	}
}

func GetSliderLayout(screenWidth, screenHeight int, index int, scrollY float32) (trackX, trackY, trackW, trackH float32) {
	panelMarginTop := TAB_BAR_HEIGHT + TOP_BAR_HEIGHT + float32(25)
	trackX = float32(screenWidth) - SLIDER_PANEL_WIDTH + 15
	trackY = panelMarginTop + float32(index)*SLIDER_SPACING + scrollY
	trackW = SLIDER_PANEL_WIDTH - 30
	trackH = 14
	return
}

// AdjustDataPoints adjusts the input vector of each DataPoint to targetInputSize.
// If targetInputSize is smaller than len(inputs), inputs are truncated.
// If targetInputSize is larger than len(inputs), inputs are padded with 0.0.
func AdjustDataPoints(points []DataPoint, targetInputSize int) []DataPoint {
	if targetInputSize <= 0 {
		return points
	}
	adjusted := make([]DataPoint, len(points))
	for i, pt := range points {
		newInputs := make([]float32, targetInputSize)
		copyLen := len(pt.inputs)
		if copyLen > targetInputSize {
			copyLen = targetInputSize
		}
		copy(newInputs, pt.inputs[:copyLen])
		adjusted[i] = DataPoint{
			inputs:  newInputs,
			outputs: pt.outputs,
		}
	}
	return adjusted
}

func ParseLayerSizes(text string, numOutputs int) ([]int, error) {
	clean := strings.ReplaceAll(text, ",", " ")
	fields := strings.Fields(clean)
	if len(fields) == 0 {
		return nil, fmt.Errorf("at least input layer size is required")
	}
	nums := make([]int, 0, len(fields)+1)
	for _, f := range fields {
		val, err := strconv.Atoi(f)
		if err != nil || val <= 0 {
			return nil, fmt.Errorf("invalid layer size: '%s'", f)
		}
		if val > 2048 {
			return nil, fmt.Errorf("layer size %d too large (max 2048)", val)
		}
		nums = append(nums, val)
	}

	// If user already specified the output layer matching numOutputs, don't duplicate it
	if len(nums) >= 2 && nums[len(nums)-1] == numOutputs {
		return nums, nil
	}
	return append(nums, numOutputs), nil
}

func SplitDataset(allPoints []DataPoint, trainPercent int) (train []DataPoint, test []DataPoint) {
	if len(allPoints) == 0 {
		return nil, nil
	}
	if trainPercent < 10 {
		trainPercent = 10
	}
	if trainPercent > 90 {
		trainPercent = 90
	}
	n := len(allPoints)
	nTrain := (n * trainPercent) / 100
	if nTrain < 1 {
		nTrain = 1
	}
	if nTrain >= n {
		nTrain = n - 1
	}
	train = make([]DataPoint, nTrain)
	test = make([]DataPoint, n-nTrain)
	copy(train, allPoints[:nTrain])
	copy(test, allPoints[nTrain:])
	return train, test
}

