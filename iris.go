package main

import (
	"strconv"
)

type IrisFlower struct {
	SepalLength float32
	SepalWidth  float32
	PetalLength float32
	PetalWidth  float32
	Species     string
}

func NewIrisFlower(sl, sw, pl, pw float32, s string) *IrisFlower {
	return &IrisFlower{
		SepalLength: sl,
		SepalWidth:  sw,
		PetalLength: pl,
		PetalWidth:  pw,
		Species:     s,
	}
}

func ReadIrisDataset(path string) []IrisFlower {
	flower_data := load_csv(path)
	var flowers []IrisFlower
	for i, row := range flower_data {
		if i == 0 {
			continue
		}
		sl, _ := strconv.ParseFloat(row[1], 32)
		sw, _ := strconv.ParseFloat(row[2], 32)
		pl, _ := strconv.ParseFloat(row[3], 32)
		pw, _ := strconv.ParseFloat(row[4], 32)
		s := row[5]
		flowers = append(flowers, *NewIrisFlower(float32(sl), float32(sw), float32(pl), float32(pw), s))
	}
	return flowers
}


func IrisToDataPoint(flowers []IrisFlower) []DataPoint {
	var dataPoints []DataPoint
	for i := 0; i < len(flowers); i += 1 {
		ip := []float32{flowers[i].PetalLength / 6.9, flowers[i].PetalWidth / 2.5, flowers[i].SepalLength / 7.9, flowers[i].SepalWidth / 4.4}
		var op []float32
		if flowers[i].Species == "Iris-setosa" {
			op = []float32{1.0, 0.0, 0.0}
		} else if flowers[i].Species == "Iris-versicolor" {
			op = []float32{0.0, 1.0, 0.0}
		} else {
			op = []float32{0.0, 0.0, 1.0}
		}
		dataPoints = append(dataPoints, DataPoint{
			inputs:  ip,
			outputs: op,
		})
	}
	return dataPoints
}
