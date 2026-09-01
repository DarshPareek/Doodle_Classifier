package main

import (
	"encoding/csv"
	"log"
	"os"
)

func load_csv(filepath string) [][]string {
	file, err := os.Open(filepath)
	if err != nil {
		log.Fatalf("Error while opening the file")
	}
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Error while reading the file")
	}
	return records
}
