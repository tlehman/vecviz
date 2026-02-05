package tsne

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
)

// EmbeddingInput represents an embedding with its prompt ID
type EmbeddingInput struct {
	ID     int64     `json:"id"`
	Vector []float32 `json:"vector"`
}

// TSNEInput is the input format for the Python script
type TSNEInput struct {
	Embeddings []EmbeddingInput `json:"embeddings"`
}

// ProjectionOutput represents a 3D projection
type ProjectionOutput struct {
	ID int64   `json:"id"`
	X  float64 `json:"x"`
	Y  float64 `json:"y"`
	Z  float64 `json:"z"`
}

// TSNEOutput is the output format from the Python script
type TSNEOutput struct {
	Projections []ProjectionOutput `json:"projections"`
}

// getProjectRoot returns the project root directory
func getProjectRoot() string {
	_, filename, _, _ := runtime.Caller(0)
	dir := filepath.Dir(filename)
	return filepath.Join(dir, "..")
}

// getScriptPath returns the path to the t-SNE Python script
func getScriptPath() string {
	return filepath.Join(getProjectRoot(), "scripts", "tsne_compute.py")
}

// ComputeTSNE runs t-SNE on the given embeddings using Python subprocess
func ComputeTSNE(embeddings []EmbeddingInput) (*TSNEOutput, error) {
	if len(embeddings) == 0 {
		return &TSNEOutput{Projections: []ProjectionOutput{}}, nil
	}

	input := TSNEInput{Embeddings: embeddings}
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	scriptPath := getScriptPath()
	cmd := exec.Command("python3", scriptPath)
	cmd.Stdin = bytes.NewReader(inputJSON)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("t-SNE failed: %v, stderr: %s", err, stderr.String())
	}

	var output TSNEOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		return nil, fmt.Errorf("failed to parse output: %w, stdout: %s", err, stdout.String())
	}

	return &output, nil
}
