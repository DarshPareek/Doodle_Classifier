package main

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

func GetSliderLayout(screenWidth, screenHeight int, index int) (trackX, trackY, trackW, trackH float32) {
	panelMarginTop := float32(35)
	sliderSpacing := float32(50)

	trackX = float32(screenWidth) - SLIDER_PANEL_WIDTH + 15
	trackY = panelMarginTop + float32(index)*sliderSpacing
	trackW = SLIDER_PANEL_WIDTH - 30
	trackH = 14
	return
}
