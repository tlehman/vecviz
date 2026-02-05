package db

import (
	"bytes"
	"database/sql"
	"encoding/binary"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

// deserializeFloat32 converts a BLOB back to []float32
func deserializeFloat32(blob []byte) ([]float32, error) {
	if len(blob)%4 != 0 {
		return nil, nil
	}
	vector := make([]float32, len(blob)/4)
	reader := bytes.NewReader(blob)
	err := binary.Read(reader, binary.LittleEndian, &vector)
	if err != nil {
		return nil, err
	}
	return vector, nil
}

var DB *sql.DB

func Init(dbPath string) error {
	sqlite_vec.Auto()

	var err error
	DB, err = sql.Open("sqlite3", dbPath)
	if err != nil {
		return err
	}

	// Create schema
	schema := `
	CREATE TABLE IF NOT EXISTS prompts (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		text TEXT NOT NULL UNIQUE,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
		prompt_id INTEGER PRIMARY KEY,
		embedding float[3072]
	);

	CREATE TABLE IF NOT EXISTS projections (
		prompt_id INTEGER PRIMARY KEY,
		x REAL NOT NULL,
		y REAL NOT NULL,
		z REAL NOT NULL,
		FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
	);
	`

	_, err = DB.Exec(schema)
	return err
}

// InsertPrompt inserts a prompt and returns its ID. If the prompt already exists, returns existing ID.
func InsertPrompt(text string) (int64, error) {
	// Check if prompt exists
	var id int64
	err := DB.QueryRow("SELECT id FROM prompts WHERE text = ?", text).Scan(&id)
	if err == nil {
		return id, nil
	}
	if err != sql.ErrNoRows {
		return 0, err
	}

	// Insert new prompt
	result, err := DB.Exec("INSERT INTO prompts (text) VALUES (?)", text)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

// InsertEmbedding stores a 3072-dim embedding for a prompt
func InsertEmbedding(promptID int64, embedding []float32) error {
	serialized, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return err
	}

	_, err = DB.Exec("INSERT INTO embeddings (prompt_id, embedding) VALUES (?, ?)", promptID, serialized)
	return err
}

// EmbeddingData holds an embedding with its prompt ID
type EmbeddingData struct {
	PromptID int64
	Vector   []float32
}

// GetAllEmbeddings retrieves all embeddings for t-SNE computation
func GetAllEmbeddings() ([]EmbeddingData, error) {
	rows, err := DB.Query("SELECT prompt_id, embedding FROM embeddings")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []EmbeddingData
	for rows.Next() {
		var promptID int64
		var blob []byte
		if err := rows.Scan(&promptID, &blob); err != nil {
			return nil, err
		}

		vector, err := deserializeFloat32(blob)
		if err != nil {
			return nil, err
		}

		results = append(results, EmbeddingData{
			PromptID: promptID,
			Vector:   vector,
		})
	}
	return results, rows.Err()
}

// Projection holds 3D coordinates for a prompt
type Projection struct {
	PromptID int64
	Text     string
	X        float64
	Y        float64
	Z        float64
}

// InsertProjections stores 3D projections (replaces existing)
func InsertProjections(projections []Projection) error {
	tx, err := DB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Clear existing projections
	_, err = tx.Exec("DELETE FROM projections")
	if err != nil {
		return err
	}

	// Insert new projections
	stmt, err := tx.Prepare("INSERT INTO projections (prompt_id, x, y, z) VALUES (?, ?, ?, ?)")
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, p := range projections {
		_, err = stmt.Exec(p.PromptID, p.X, p.Y, p.Z)
		if err != nil {
			return err
		}
	}

	return tx.Commit()
}

// GetAllProjections retrieves all 3D projections with prompt text
func GetAllProjections() ([]Projection, error) {
	rows, err := DB.Query(`
		SELECT p.prompt_id, pr.text, p.x, p.y, p.z
		FROM projections p
		JOIN prompts pr ON p.prompt_id = pr.id
		ORDER BY p.prompt_id
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Projection
	for rows.Next() {
		var p Projection
		if err := rows.Scan(&p.PromptID, &p.Text, &p.X, &p.Y, &p.Z); err != nil {
			return nil, err
		}
		results = append(results, p)
	}
	return results, rows.Err()
}

// GetEmbeddingCount returns the number of stored embeddings
func GetEmbeddingCount() (int, error) {
	var count int
	err := DB.QueryRow("SELECT COUNT(*) FROM embeddings").Scan(&count)
	return count, err
}

// GetProjectionCount returns the number of stored projections
func GetProjectionCount() (int, error) {
	var count int
	err := DB.QueryRow("SELECT COUNT(*) FROM projections").Scan(&count)
	return count, err
}
