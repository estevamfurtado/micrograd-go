package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

const (
	trainURL = "https://pjreddie.com/media/files/mnist_train.csv"
	testURL  = "https://pjreddie.com/media/files/mnist_test.csv"
)

var dataFiles = []struct {
	url  string
	name string
}{
	{trainURL, "mnist_train.csv"},
	{testURL, "mnist_test.csv"},
}

func dataDir() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "data"
	}
	return filepath.Join(filepath.Dir(file), "data")
}

func ensureData() error {
	dir := dataDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	for _, f := range dataFiles {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		fmt.Printf("downloading %s …\n", f.url)
		if err := downloadFile(f.url, path); err != nil {
			return err
		}
		fmt.Printf("saved %s\n", path)
	}
	return nil
}

func downloadFile(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	tmp := path + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}

	_, err = io.Copy(out, resp.Body)
	closeErr := out.Close()
	if err != nil {
		os.Remove(tmp)
		return err
	}
	if closeErr != nil {
		os.Remove(tmp)
		return closeErr
	}

	return os.Rename(tmp, path)
}
