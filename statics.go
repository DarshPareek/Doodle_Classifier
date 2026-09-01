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

const (
	BOUNDARY_RESOLUTION = 500
	SLIDER_PANEL_WIDTH  = float32(200)
	DEFAULT_WINDOW_W    = 1000
	DEFAULT_WINDOW_H    = 800
)
