package main


func ParseNN(nn *NeuralNetwork) []Slider {
	var sliders []Slider
	for i, layer := range nn.layers {
		for j := 0; j < len(layer.weights); j++ {
			for k := 0; k < len(layer.weights[j]); k++ {
				s := NewSlider("Weight", i, j, k, 0)
				s.SetValue(nn.layers[i].weights[j][k])
				sliders = append(sliders, *s)
			}
		}
		for j := 0; j < len(layer.biases); j++ {
			s := NewSlider("Bias", i, 0, 0, j)
			s.SetValue(nn.layers[i].biases[j])
			sliders = append(sliders, *s)
		}
	}
	return sliders
}
