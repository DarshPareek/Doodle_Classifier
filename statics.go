package main

import "image/color"

var BG_COLOR = color.RGBA{
	R: 26,
	G: 27,
	B: 28,
	A: 255,
}

var LINE_COLOR = color.RGBA{
	R: 103,
	G: 103,
	B: 103,
	A: 255,
}

var COLOR_1 = color.RGBA{
	//rgb(91 175 252)
	R: 91,
	G: 175,
	B: 252,
	A: 255,
}

var COLOR_2 = color.RGBA{
	//rgb(253 79 86)
	R: 253,
	G: 79,
	B: 86,
	A: 255,
}

var COLOR_3 = color.RGBA{
	R: 120,
	G: 120,
	B: 130,
	A: 255,
}

var COLOR_4 = color.RGBA{
	R: 18,
	G: 19,
	B: 20,
	A: 255,
}

var CLASS_COLORS = []color.RGBA{
	COLOR_1,                         // Class 0: Blue rgb(91, 175, 252)
	COLOR_2,                         // Class 1: Red/Pink rgb(253, 79, 86)
	{R: 76, G: 175, B: 80, A: 255},  // Class 2: Green
	{R: 255, G: 167, B: 38, A: 255}, // Class 3: Orange
	{R: 171, G: 71, B: 188, A: 255}, // Class 4: Purple
	{R: 0, G: 188, B: 212, A: 255},  // Class 5: Cyan
	{R: 255, G: 214, B: 0, A: 255},  // Class 6: Yellow
	{R: 233, G: 30, B: 99, A: 255},  // Class 7: Pink
	{R: 139, G: 195, B: 74, A: 255}, // Class 8: Light Green
	{R: 255, G: 87, B: 34, A: 255},  // Class 9: Deep Orange
}

func GetClassColor(index int) color.RGBA {
	if index >= 0 && index < len(CLASS_COLORS) {
		return CLASS_COLORS[index]
	}
	return COLOR_3
}

const (
	BOUNDARY_RESOLUTION      = 150
	SLIDER_PANEL_WIDTH       = float32(220)
	DEFAULT_WINDOW_W         = 1000
	DEFAULT_WINDOW_H         = 800
	EPOCHS_PER_FRAME         = 2
	BOUNDARY_UPDATE_INTERVAL = 6
	TOP_BAR_HEIGHT           = float32(40)
	SLIDER_SPACING           = float32(48)
	TAB_BAR_HEIGHT           = float32(36)
	TAB_WIDTH                = float32(180)
)

var WEIGHT_RANGE_MULTIPLIER = float32(5.0)
var WEIGHT_RANGE_ADDER = float32(2.5)
