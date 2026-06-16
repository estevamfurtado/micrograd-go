package sample

import (
	"path/filepath"
	"runtime"
)

func packageDir() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	return filepath.Dir(file)
}

func DataDir() string {
	return filepath.Join(packageDir(), "data")
}

func dataPath(name string) string {
	return filepath.Join(DataDir(), name)
}
