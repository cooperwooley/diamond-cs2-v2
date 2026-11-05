package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"testing"
)

// The Python cache key includes behaviorVersion, so a change to what the tables contain must bump it. This test pins
// the version together with a hash of the files that decide the contents. When it fails after an intentional
// change: bump behaviorVersion in main.go (and DEMODUMP_BEHAVIOR in cs2wm/demodump.py), then update both values
// below.
const (
	pinnedBehaviorVersion = "1"
	pinnedSourceHash      = "f2a493ab33fd6c300363412d1320552a16349f3ad73e59c6d64113fff8890299"
)

func sourceHash(t *testing.T) string {
	t.Helper()
	h := sha256.New()
	for _, name := range []string{"derive.go", "rows.go", "tables.go"} {
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		h.Write(bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))) // a Windows checkout may have CRLF
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestBehaviorVersionIsBumpedWithTheSources(t *testing.T) {
	if behaviorVersion != pinnedBehaviorVersion {
		t.Fatalf("behaviorVersion is %q but the test pins %q: update pinnedBehaviorVersion too", behaviorVersion, pinnedBehaviorVersion)
	}
	if got := sourceHash(t); got != pinnedSourceHash {
		t.Fatalf("derive.go, rows.go or tables.go changed (hash %s): bump behaviorVersion, then set pinnedBehaviorVersion and pinnedSourceHash", got)
	}
}
