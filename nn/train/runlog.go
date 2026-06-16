package train

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type RunLogger struct {
	dir       string
	startTime time.Time
	config    Config
	epochs    []runEpochRow
	epochLogs []runEpochLog
}

type runEpochLog struct {
	epoch   int
	batches []runBatchRow
}

type runEpochRow struct {
	epoch     int
	lr        string
	testAcc   float64
	trainAcc  float64
}

type runBatchRow struct {
	batch    int
	minutes  float64
	loss     float64
	accuracy float64
}

func NewRunLogger(baseDir string, config Config) (*RunLogger, error) {
	ts := time.Now().Format("2006-01-02_15-04-05")
	dir := filepath.Join(baseDir, ts)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &RunLogger{
		dir:       dir,
		startTime: time.Now(),
		config:    config,
	}, nil
}

func (r *RunLogger) Dir() string {
	return r.dir
}

func (r *RunLogger) RecordInitial(testAccuracy float64) error {
	r.epochs = append(r.epochs, runEpochRow{
		epoch:    -1,
		lr:       "-",
		testAcc:  testAccuracy,
		trainAcc: testAccuracy,
	})
	return r.writeSummary()
}

func (r *RunLogger) RecordEpoch(epoch int, lr, testAccuracy, trainAccuracy float64, batches []runBatchRow) error {
	r.epochs = append(r.epochs, runEpochRow{
		epoch:    epoch,
		lr:       fmt.Sprintf("%.3g", lr),
		testAcc:  testAccuracy,
		trainAcc: trainAccuracy,
	})
	r.epochLogs = append(r.epochLogs, runEpochLog{epoch: epoch, batches: batches})

	if err := r.writeEpochFile(epoch, batches); err != nil {
		return err
	}
	if err := r.writeLogs(); err != nil {
		return err
	}
	return r.writeSummary()
}

func (r *RunLogger) writeSummary() error {
	var b strings.Builder
	r.writeConfigHeader(&b)
	b.WriteString("\n")
	r.writeEpochTable(&b, r.epochs)
	return os.WriteFile(filepath.Join(r.dir, "summary.txt"), []byte(b.String()), 0o644)
}

func (r *RunLogger) writeEpochFile(epoch int, batches []runBatchRow) error {
	var b strings.Builder
	r.writeBatchTable(&b, batches)
	path := filepath.Join(r.dir, fmt.Sprintf("epoch_%d.txt", epoch))
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func (r *RunLogger) writeLogs() error {
	var b strings.Builder
	r.writeConfigHeader(&b)
	b.WriteString("\n\n")
	r.writeEpochTable(&b, r.epochs)
	b.WriteString("\n")

	for _, section := range r.epochLogs {
		fmt.Fprintf(&b, "Epoch %d\n", section.epoch)
		r.writeBatchTable(&b, section.batches)
		b.WriteString("\n")
	}

	return os.WriteFile(filepath.Join(r.dir, "logs.txt"), []byte(b.String()), 0o644)
}

func (r *RunLogger) writeConfigHeader(b *strings.Builder) {
	c := r.config
	fmt.Fprintf(b, "config {\n")
	fmt.Fprintf(b, "  hidden_size: %d\n", c.HiddenSize)
	fmt.Fprintf(b, "  limit: %d\n", c.Limit)
	fmt.Fprintf(b, "  test_limit: %d\n", c.TestLimit)
	fmt.Fprintf(b, "  epochs: %d\n", c.Epochs)
	fmt.Fprintf(b, "  batch_size: %d\n", c.BatchSize)
	fmt.Fprintf(b, "  lr: %g\n", c.LR)
	fmt.Fprintf(b, "}\n\n")
	fmt.Fprintf(b, "start_time: %s\n", r.startTime.Format("2006-01-02 15:04:05"))
}

func (r *RunLogger) writeEpochTable(b *strings.Builder, rows []runEpochRow) {
	b.WriteString("| Epoch | Learning rate | Test accuracy | Train accuracy |\n")
	b.WriteString("|-------|---------------|---------------|----------------|\n")
	for _, row := range rows {
		fmt.Fprintf(b, "| %-5d | %-13s | %-13.2f | %-14.2f |\n",
			row.epoch, row.lr, row.testAcc, row.trainAcc)
	}
}

func (r *RunLogger) writeBatchTable(b *strings.Builder, batches []runBatchRow) {
	b.WriteString("| Batch | min | accuracy | Loss |\n")
	b.WriteString("|-------|-----|----------|------|\n")
	for _, row := range batches {
		label := fmt.Sprintf("%d", row.batch)
		fmt.Fprintf(b, "| %-5s | %.2f | %.2f | %.2f |\n",
			label, row.minutes, row.accuracy, row.loss)
	}
	if len(batches) > 0 {
		last := batches[len(batches)-1]
		fmt.Fprintf(b, "| %-5s | %.2f | %.2f | %.2f |\n",
			"Final", last.minutes, last.accuracy, last.loss)
	}
}
