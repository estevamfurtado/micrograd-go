package datasets

import (
	"encoding/json"
	"os"
)

// WriteJSONL writes one JSON object per line: {"x":[x0,x1],"y":label}.
func WriteJSONL(path string, samples []Sample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, s := range samples {
		if err := enc.Encode(s); err != nil {
			return err
		}
	}
	return nil
}
