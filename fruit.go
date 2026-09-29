package main

import (
	"strconv"
)

type Fruit struct {
	SpikeLength float32
	SpotSize    float32
	Poisonous   bool
}

func NewFruit(sl, ss float32, p bool) *Fruit {
	return &Fruit{
		SpikeLength: sl,
		SpotSize:    ss,
		Poisonous:   p,
	}
}

func ReadFruitDataset(path string) []Fruit {
	fruit_data := load_csv(path)
	var fruits []Fruit
	for i, row := range fruit_data {
		if i == 0 {
			continue
		}
		sl, _ := strconv.ParseFloat(row[1], 32)
		ss, _ := strconv.ParseFloat(row[2], 32)
		p, _ := strconv.ParseBool(row[3])
		fruits = append(fruits, *NewFruit(float32(sl), float32(ss), p))
	}
	return fruits
}

func FruitToDataPoint(fruits []Fruit) []DataPoint {
	var dataPoints []DataPoint
	for i := 0; i < len(fruits); i += 1 {
		ip := []float32{fruits[i].SpikeLength / 8.0, fruits[i].SpotSize / 5.0}
		var op []float32
		if fruits[i].Poisonous == true {
			op = []float32{1.0, 0.0}
		} else {
			op = []float32{0.0, 1.0}
		}
		dataPoints = append(dataPoints, DataPoint{
			inputs:  ip,
			outputs: op,
		})
	}
	return dataPoints
}
