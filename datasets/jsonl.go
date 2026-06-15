package datasets

import (
	"encoding/json"
	"io"
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

// ReadJSONL reads one JSON object per line into samples.
func ReadJSONL(path string) (Samples, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	samples := []Sample{}

	for {
		var s Sample
		err := dec.Decode(&s)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		samples = append(samples, s)
	}

	return samples, nil
}
