package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunLogger_writesSummaryAndEpochFiles(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		HiddenSize: 16,
		Limit:      100,
		TestLimit:  20,
		Epochs:     2,
		BatchSize:  32,
		LR:         0.02,
	}

	logger, err := NewRunLogger(dir, cfg)
	require.NoError(t, err)

	require.NoError(t, logger.RecordInitial(0.12))
	require.NoError(t, logger.RecordEpoch(0, 0.02, 0.15, 0.18, []runBatchRow{
		{batch: 0, minutes: 0.01, loss: 2.3, accuracy: 0.10},
	}))

	summary, err := os.ReadFile(filepath.Join(logger.Dir(), "summary.txt"))
	require.NoError(t, err)
	body := string(summary)
	assert.Contains(t, body, "hidden_size: 16")
	assert.Contains(t, body, "| -1    | -")
	assert.Contains(t, body, "| 0     | 0.02")

	epoch, err := os.ReadFile(filepath.Join(logger.Dir(), "epoch_0.txt"))
	require.NoError(t, err)
	assert.True(t, strings.Contains(string(epoch), "| Final"))

	logs, err := os.ReadFile(filepath.Join(logger.Dir(), "logs.txt"))
	require.NoError(t, err)
	assert.Contains(t, string(logs), "Epoch 0")
}
