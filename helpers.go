package main

func LoadIrisSepalPoints(data []IrisFlower, num int) []Point {
	var res []Point
	for i, flower := range data {
		if i == num {
			break
		}
		res = append(res, Point{
			x:    uint(flower.SepalLength * 10),
			y:    uint(flower.PetalLength * 10),
			flag: 0,
		})
	}
	return res
}
func LoadIrisPetalPoints(data []IrisFlower, num int) []Point {
	var res []Point
	for i, flower := range data {
		if i == num {
			break
		}
		res = append(res, Point{
			x:    uint(flower.SepalWidth * 10),
			y:    uint(flower.PetalWidth * 10),
			flag: 1,
		})
	}
	return res
}

func ParseNN(nn *NeuralNetwork) []Slider {
	var sliders []Slider
	for i, layer := range nn.layers {
		for j := 0; j < len(layer.weights); j++ {
			for k := 0; k < len(layer.weights[0]); k++ {
				s := NewSlider("Weight", i, j, k, 0)
				sliders = append(sliders, *s)
				nn.layers[i].weights[j][k] = s.Value()
			}
		}
		for j := 0; j < len(layer.biases); j++ {
			s := NewSlider("Bias", i, 0, 0, j)
			sliders = append(sliders, *s)
			nn.layers[i].biases[j] = s.Value()
		}
	}
	return sliders
}
