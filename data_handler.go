package main

import (
	"encoding/csv"
	"log"
	"os"
	"path/filepath"
)

func load_csv(filePath string) [][]string {
	file, err := os.Open(filePath)
	if err != nil {
		alt := filepath.Join("archive", "iris", filepath.Base(filePath))
		if f2, err2 := os.Open(alt); err2 == nil {
			file = f2
		} else {
			log.Fatalf("Error while opening file %s: %v", filePath, err)
		}
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Error while reading file %s: %v", filePath, err)
	}
	return records
}
