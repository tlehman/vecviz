package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

const (
	DefaultBaseURL = "http://localhost:11434"
	Model          = "llama3.2"
)

type Client struct {
	baseURL string
	http    *http.Client
}

func NewClient(baseURL string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		baseURL: baseURL,
		http:    &http.Client{},
	}
}

type embedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// GetEmbedding calls the Ollama embed API and returns the embedding vector
func (c *Client) GetEmbedding(text string) ([]float32, error) {
	reqBody := embedRequest{
		Model: Model,
		Input: text,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.http.Post(c.baseURL+"/api/embed", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to call ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	var embedResp embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// Convert float64 to float32
	embedding := make([]float32, len(embedResp.Embeddings[0]))
	for i, v := range embedResp.Embeddings[0] {
		embedding[i] = float32(v)
	}

	return embedding, nil
}
