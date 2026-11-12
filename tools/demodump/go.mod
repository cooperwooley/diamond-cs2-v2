module github.com/cooperwooley/diamond-cs2-v2/tools/demodump

go 1.27.3

require github.com/markus-wa/demoinfocs-golang/v6 v6.0.0-rc.1

require (
	github.com/golang/geo v0.0.0-20230421003525-6adc56603217 // indirect
	github.com/golang/snappy v1.0.0 // indirect
	github.com/markus-wa/go-unassert v0.1.3 // indirect
	github.com/markus-wa/gobitread v0.2.5 // indirect
	github.com/markus-wa/godispatch v1.5.1 // indirect
	github.com/markus-wa/ice-cipher-go v1.0.0 // indirect
	github.com/markus-wa/quickhull-go/v2 v2.2.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)

// CS2's 2026-11-11 update sends some uint8 fields with the fixed8 encoder, which v6.0.0-rc.1 reads as a varint, so
// every new demo fails early with ErrDecodeFailed "insufficient buffer". Upstream PR markus-wa/demoinfocs-golang#764
// fixes it; this pins that PR's commit (v6.0.0-rc.1 plus the fix, nothing else) until a release has it.
replace github.com/markus-wa/demoinfocs-golang/v6 => github.com/GamersClub/demoinfocs-golang/v6 v6.0.0-20261112032156-6d7a0a27e601
