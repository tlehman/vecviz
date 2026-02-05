package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/tlehman/vecviz/db"
	"github.com/tlehman/vecviz/ollama"
	"github.com/tlehman/vecviz/tsne"
)

var ollamaClient *ollama.Client

func main() {
	// Initialize database
	if err := db.Init("vecviz.db"); err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	log.Println("Database initialized")

	// Initialize Ollama client
	ollamaClient = ollama.NewClient("")

	// Set up routes
	http.HandleFunc("/embed", handleEmbed)
	http.HandleFunc("/tsne/compute", handleTSNECompute)
	http.HandleFunc("/points", handlePoints)
	http.Handle("/", http.FileServer(http.Dir("static")))

	log.Println("Server starting on http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

// POST /embed - Add a new embedding
func handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "Prompt is required", http.StatusBadRequest)
		return
	}

	// Check if prompt already exists
	existingID, _ := db.InsertPrompt(req.Prompt)

	// Check if embedding already exists for this prompt
	embedCount, _ := db.GetEmbeddingCount()
	projCount, _ := db.GetProjectionCount()

	// Get embedding from Ollama
	embedding, err := ollamaClient.GetEmbedding(req.Prompt)
	if err != nil {
		log.Printf("Ollama error: %v", err)
		http.Error(w, "Failed to get embedding: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Store embedding
	if err := db.InsertEmbedding(existingID, embedding); err != nil {
		// Might already exist, which is fine
		log.Printf("Insert embedding: %v", err)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":                existingID,
		"prompt":            req.Prompt,
		"embedding_dim":     len(embedding),
		"needs_tsne_update": embedCount != projCount,
	})
}

// POST /tsne/compute - Recompute t-SNE projections
func handleTSNECompute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	// Get all embeddings
	embeddings, err := db.GetAllEmbeddings()
	if err != nil {
		http.Error(w, "Failed to get embeddings: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if len(embeddings) == 0 {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":              "completed",
			"points_processed":    0,
			"computation_time_ms": 0,
		})
		return
	}

	// Convert to t-SNE input format
	tsneInput := make([]tsne.EmbeddingInput, len(embeddings))
	for i, e := range embeddings {
		tsneInput[i] = tsne.EmbeddingInput{
			ID:     e.PromptID,
			Vector: e.Vector,
		}
	}

	// Run t-SNE
	output, err := tsne.ComputeTSNE(tsneInput)
	if err != nil {
		log.Printf("t-SNE error: %v", err)
		http.Error(w, "t-SNE failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Store projections
	projections := make([]db.Projection, len(output.Projections))
	for i, p := range output.Projections {
		projections[i] = db.Projection{
			PromptID: p.ID,
			X:        p.X,
			Y:        p.Y,
			Z:        p.Z,
		}
	}

	if err := db.InsertProjections(projections); err != nil {
		http.Error(w, "Failed to store projections: "+err.Error(), http.StatusInternalServerError)
		return
	}

	elapsed := time.Since(start)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":              "completed",
		"points_processed":    len(projections),
		"computation_time_ms": elapsed.Milliseconds(),
	})
}

// GET /points - Get all 3D projections
func handlePoints(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	projections, err := db.GetAllProjections()
	if err != nil {
		http.Error(w, "Failed to get projections: "+err.Error(), http.StatusInternalServerError)
		return
	}

	embedCount, _ := db.GetEmbeddingCount()
	projCount, _ := db.GetProjectionCount()

	points := make([]map[string]interface{}, len(projections))
	for i, p := range projections {
		points[i] = map[string]interface{}{
			"id":   p.PromptID,
			"text": p.Text,
			"x":    p.X,
			"y":    p.Y,
			"z":    p.Z,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"points":       points,
		"needs_update": embedCount != projCount,
	})
}
