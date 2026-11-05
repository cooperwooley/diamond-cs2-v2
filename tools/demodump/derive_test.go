package main

import (
	"reflect"
	"strconv"
	"testing"
)

// run feeds a sequence of samples to a fresh detector and returns "tick:name" for every derived event.
func run(samples []sample) []string {
	var d detector
	var got []string
	for _, s := range samples {
		for _, e := range d.step(s) {
			got = append(got, strconv.Itoa(e.Tick)+":"+e.Name)
		}
	}
	return got
}

// ground returns n grounded samples starting at tick t0 holding a rifle (entity 7, 30 rounds).
func ground(t0, n int) []sample {
	out := make([]sample, n)
	for i := range out {
		out[i] = sample{Tick: t0 + i, Alive: true, Z: 10, Weapon: 7, WeapName: "AK-47", Clip: 30}
	}
	return out
}

func TestJumpIsReportedAtTakeoffTick(t *testing.T) {
	s := ground(100, 3)
	for i, z := range []float64{14.7, 24.0, 32.0} {
		s = append(s, sample{Tick: 103 + i, Alive: true, Z: 10 + (z - 10), Airborne: true, Weapon: 7, Clip: 30})
	}
	if got, want := run(s), []string{"103:player_jump"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestJumpRiseMayArriveAfterTheEdge(t *testing.T) {
	s := ground(100, 2)
	s = append(s,
		sample{Tick: 102, Alive: true, Z: 10, Airborne: true, Weapon: 7, Clip: 30}, // snapshot not yet updated
		sample{Tick: 103, Alive: true, Z: 10, Airborne: true, Weapon: 7, Clip: 30},
		sample{Tick: 104, Alive: true, Z: 22, Airborne: true, Weapon: 7, Clip: 30},
	)
	if got, want := run(s), []string{"102:player_jump"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestFallingOffALedgeIsNotAJump(t *testing.T) {
	s := ground(100, 2)
	for i, z := range []float64{9.8, 9.0, 7.0, 3.0, -4.0, -14.0} {
		s = append(s, sample{Tick: 102 + i, Alive: true, Z: z, Airborne: true, Weapon: 7, Clip: 30})
	}
	if got := run(s); len(got) != 0 {
		t.Fatalf("got %v want none", got)
	}
}

func TestRiseAfterTheWindowIsNotAJump(t *testing.T) {
	s := ground(100, 2)
	for i := 0; i < jumpWindowTicks+2; i++ {
		s = append(s, sample{Tick: 102 + i, Alive: true, Z: 10, Airborne: true, Weapon: 7, Clip: 30})
	}
	s = append(s, sample{Tick: 102 + jumpWindowTicks + 2, Alive: true, Z: 30, Airborne: true, Weapon: 7, Clip: 30})
	if got := run(s); len(got) != 0 {
		t.Fatalf("got %v want none", got)
	}
}

func TestRespawnTeleportIsNotAJump(t *testing.T) {
	s := []sample{
		{Tick: 0, Alive: true, Z: -1612.47, Airborne: true},
		{Tick: 1, Alive: true, Z: 8.0, Airborne: false},
		{Tick: 2, Alive: true, Z: 8.0, Airborne: false},
	}
	if got := run(s); len(got) != 0 {
		t.Fatalf("got %v want none", got)
	}
	dead := []sample{
		{Tick: 0, Alive: false, Z: 0, Airborne: true},
		{Tick: 1, Alive: true, Z: 300, Airborne: true},
		{Tick: 2, Alive: true, Z: 330, Airborne: true},
	}
	if got := run(dead); len(got) != 0 {
		t.Fatalf("got %v want none", got)
	}
}

func TestReloadFlagEdgeAndRefillAreReportedOnce(t *testing.T) {
	s := ground(100, 2)
	s[1].Clip = 12
	flag := sample{Tick: 102, Alive: true, Z: 10, Weapon: 7, Clip: 12, InReload: true}
	s = append(s, flag)
	flag.Tick, s = 103, append(s, flag)
	flag.Tick, flag.Clip, flag.InReload = 200, 30, false // refill at the end of the reload, inside the debounce
	s = append(s, flag)
	if got, want := run(s), []string{"102:weapon_reload"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestReloadFromClipRefillWhenTheFlagIsMissing(t *testing.T) {
	s := ground(100, 2)
	s[1].Clip = 5
	s = append(s, sample{Tick: 102, Alive: true, Z: 10, Weapon: 7, Clip: 30})
	if got, want := run(s), []string{"102:weapon_reload"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestWeaponSwitchIsNeitherReloadNorZoom(t *testing.T) {
	s := ground(100, 2)
	s[0].Zoom, s[1].Clip, s[1].Zoom = 2, 5, 2
	s = append(s, sample{Tick: 102, Alive: true, Z: 10, Weapon: 9, Clip: 30, Zoom: 0, InReload: true})
	if got := run(s); len(got) != 0 {
		t.Fatalf("got %v want none", got)
	}
}

func TestZoomChangesAreReportedButNotTheUnscopeAfterAShot(t *testing.T) {
	awp := func(tick, clip, zoom int) sample {
		return sample{Tick: tick, Alive: true, Z: 10, Weapon: 5, Clip: clip, Zoom: zoom}
	}
	s := []sample{awp(100, 5, 0), awp(101, 5, 0), awp(102, 5, 1), awp(110, 5, 2), awp(120, 5, 0), awp(130, 5, 2), awp(140, 4, 0)}
	want := []string{"102:weapon_zoom", "110:weapon_zoom", "120:weapon_zoom", "130:weapon_zoom"}
	if got := run(s); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
