package main

import (
	"math"
	"strconv"
)

// Placeholder pose: until a player's first real snapshot arrives (demo tick 0 up to the spawn), the pawn sits at
// (0, 0, -1612.47) with zero angles. Those rows carry no information and are written with an empty pose.
const (
	placeholderZ   = -1612.47
	placeholderTol = 0.01
)

// playerState is everything one ticks.csv.gz row holds, before formatting.
type playerState struct {
	Tick, UserID, EntityID, PawnID, Team, Health int
	Name                                         string
	SteamID                                      uint64
	HasPawn, Alive                               bool
	X, Y, Z, Yaw, Pitch, Duck                    float64
	Airborne, Walking, Scoped                    bool
	Weapon                                       string
}

func isPlaceholder(x, y, z float64) bool {
	return x == 0 && y == 0 && math.Abs(z-placeholderZ) < placeholderTol
}

// validPose reports whether the pose columns mean anything: there is a pawn and it is not at the placeholder.
func (s playerState) validPose() bool {
	return s.HasPawn && !isPlaceholder(s.X, s.Y, s.Z)
}

// alive is false for a player without a pawn, whatever the controller says.
func (s playerState) alive() bool { return s.HasPawn && s.Alive }

// cells formats the row in tickColumns order. Without a valid pose, X, Y, Z, yaw, pitch, duck_amount and the weapon
// are empty (NaN in pandas) and the boolean pose flags are false.
func (s playerState) cells() []string {
	pose := s.validPose()
	num := func(v float64, prec int) string {
		if !pose {
			return ""
		}
		return strconv.FormatFloat(v, 'f', prec, 64)
	}
	flag := func(v bool) string { return strconv.FormatBool(pose && v) }
	weapon := s.Weapon
	if !pose {
		weapon = ""
	}
	return []string{
		strconv.Itoa(s.Tick), strconv.Itoa(s.UserID), strconv.Itoa(s.EntityID), strconv.Itoa(s.PawnID), s.Name,
		strconv.FormatUint(s.SteamID, 10), strconv.Itoa(s.Team), strconv.FormatBool(s.alive()), strconv.Itoa(s.Health),
		num(s.X, 3), num(s.Y, 3), num(s.Z, 3), num(s.Yaw, 4), num(s.Pitch, 4),
		flag(s.Airborne), num(s.Duck, 4), weapon, flag(s.Walking), flag(s.Scoped),
	}
}
