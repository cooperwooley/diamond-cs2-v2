package main

import (
	"reflect"
	"testing"
)

func column(t *testing.T, name string, cells []string) string {
	t.Helper()
	if len(cells) != len(tickColumns) {
		t.Fatalf("%d cells for %d columns", len(cells), len(tickColumns))
	}
	for i, c := range tickColumns {
		if c == name {
			return cells[i]
		}
	}
	t.Fatalf("no column %s", name)
	return ""
}

func alive() playerState {
	return playerState{
		Tick: 500, UserID: 3, EntityID: 4, PawnID: 339, Team: 2, Health: 100, Name: "Wesley",
		HasPawn: true, Alive: true, X: 175.06, Y: 75.38, Z: 8.03, Yaw: 12.0045, Pitch: 5.7351,
		Airborne: true, Walking: true, Weapon: "AK-47",
	}
}

func TestValidRowKeepsEveryColumn(t *testing.T) {
	c := alive().cells()
	want := map[string]string{
		"X": "175.060", "Z": "8.030", "yaw": "12.0045", "pitch": "5.7351", "is_alive": "true",
		"is_airborne": "true", "is_walking": "true", "active_weapon_name": "AK-47", "duck_amount": "0.0000",
	}
	for name, v := range want {
		if got := column(t, name, c); got != v {
			t.Errorf("%s = %q want %q", name, got, v)
		}
	}
}

func TestRowWithoutPawnHasNoPoseAndIsDead(t *testing.T) {
	s := alive()
	s.HasPawn, s.PawnID = false, 0
	c := s.cells()
	for _, name := range []string{"X", "Y", "Z", "yaw", "pitch", "duck_amount", "active_weapon_name"} {
		if got := column(t, name, c); got != "" {
			t.Errorf("%s = %q want empty", name, got)
		}
	}
	if got := column(t, "is_alive", c); got != "false" {
		t.Errorf("is_alive = %q want false", got)
	}
	if got := column(t, "is_airborne", c); got != "false" {
		t.Errorf("is_airborne = %q want false", got)
	}
	if got := column(t, "health", c); got != "100" {
		t.Errorf("health is not pose and stays, got %q", got)
	}
}

func TestPlaceholderPoseIsBlankedButAliveIsKept(t *testing.T) {
	s := alive()
	s.X, s.Y, s.Z, s.Yaw, s.Pitch = 0, 0, -1612.469, 0, 0
	c := s.cells()
	if got := column(t, "Z", c); got != "" {
		t.Errorf("Z = %q want empty", got)
	}
	if got := column(t, "is_alive", c); got != "true" {
		t.Errorf("is_alive = %q want true", got)
	}
	// A genuine position that merely has x = y = 0 is not the placeholder.
	s.Z = 8
	if got := column(t, "Z", s.cells()); got != "8.000" {
		t.Errorf("Z = %q want 8.000", got)
	}
}

func TestIsPlaceholder(t *testing.T) {
	cases := []struct {
		x, y, z float64
		want    bool
	}{
		{0, 0, -1612.47, true}, {0, 0, -1612.469, true}, {0, 0, -1600, false}, {1, 0, -1612.47, false}, {0, 0, 0, false},
	}
	for _, c := range cases {
		if got := isPlaceholder(c.x, c.y, c.z); got != c.want {
			t.Errorf("isPlaceholder(%v, %v, %v) = %v", c.x, c.y, c.z, got)
		}
	}
}

func TestOccupancyOpensANewIntervalWhenTheUserIDIsReused(t *testing.T) {
	d := &dumper{current: map[int]*occupancy{}}
	at := func(tick int, name string, steam uint64, entity int) {
		s := alive()
		s.Tick, s.Name, s.SteamID, s.EntityID = tick, name, steam, entity
		d.occupy(s, false)
	}
	at(10, "Ann", 1, 4)
	at(11, "Ann", 1, 4)
	at(500, "Bob", 2, 7) // same user_id 3, new player
	at(501, "Bob", 2, 7)
	var got [][3]any
	for _, o := range d.roster {
		got = append(got, [3]any{o.name, o.first, o.last})
	}
	want := [][3]any{{"Ann", 10, 11}, {"Bob", 500, 501}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
