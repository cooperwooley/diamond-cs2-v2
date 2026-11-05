package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// demoFromEnv returns the demo named by DCS2_TEST_DEM (a real CS2 demo is too big to check in) or skips.
func demoFromEnv(t *testing.T) []byte {
	t.Helper()
	path := os.Getenv("DCS2_TEST_DEM")
	if path == "" {
		t.Skip("set DCS2_TEST_DEM to a .dem file to run this test")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func readHeader(t *testing.T, dir string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "header.json"))
	if err != nil {
		t.Fatal(err)
	}
	var h map[string]any
	if err := json.Unmarshal(b, &h); err != nil {
		t.Fatal(err)
	}
	return h
}

func TestTruncatedDemoGivesAPartialDump(t *testing.T) {
	full := demoFromEnv(t)
	tmp := t.TempDir()
	whole, cut := filepath.Join(tmp, "whole.dem"), filepath.Join(tmp, "cut.dem")
	if err := os.WriteFile(whole, full, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(cut, full[:len(full)/2], 0o644); err != nil {
		t.Fatal(err)
	}

	if err := dump(whole, filepath.Join(tmp, "whole")); err != nil {
		t.Fatalf("full demo: %v", err)
	}
	if h := readHeader(t, filepath.Join(tmp, "whole")); h["truncated"] != false {
		t.Fatalf("full demo: truncated = %v", h["truncated"])
	}

	if err := dump(cut, filepath.Join(tmp, "cut")); err != nil {
		t.Fatalf("truncated demo should give a partial dump, got %v", err)
	}
	hw, hc := readHeader(t, filepath.Join(tmp, "whole")), readHeader(t, filepath.Join(tmp, "cut"))
	if hc["truncated"] != true {
		t.Fatalf("truncated = %v", hc["truncated"])
	}
	if lc, lw := hc["last_tick"].(float64), hw["last_tick"].(float64); lc <= 0 || lc >= lw {
		t.Fatalf("last_tick of the partial dump %v should be inside (0, %v)", lc, lw)
	}
	for _, name := range []string{"ticks.csv.gz", "players.csv", "grenades.csv.gz"} {
		if _, err := os.Stat(filepath.Join(tmp, "cut", name)); err != nil {
			t.Errorf("partial dump lacks %s: %v", name, err)
		}
	}
}

func TestGarbageIsAHardFailure(t *testing.T) {
	tmp := t.TempDir()
	bad := filepath.Join(tmp, "bad.dem")
	if err := os.WriteFile(bad, []byte("this is not a demo at all, not even close"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := dump(bad, filepath.Join(tmp, "out")); err == nil {
		t.Fatal("expected an error for a file that is not a demo")
	}
	if _, err := os.Stat(filepath.Join(tmp, "out", "header.json")); err == nil {
		t.Fatal("a failed dump must not write header.json")
	}
}
