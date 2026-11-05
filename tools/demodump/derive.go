package main

// Derived events: GOTV demos carry no player_jump, weapon_reload or weapon_zoom game events (and no grenade_thrown),
// so they are reconstructed from the per-tick player state. Each is written with derived=true. A derived event is
// only kept when the demo has no raw event of the same name (client demos do).

// Thresholds, tuned for 64 Hz. They are constants so that a change shows up in review and in the tests.
const (
	// jumpMinRise is the height (Source units) a player must gain above the last grounded Z, within
	// jumpWindowTicks of leaving the ground, for the take-off to count as a jump. A jump launches at 301 u/s, i.e.
	// about 4.7 u per tick, so a real jump clears this within two ticks. Walking off a ledge or a ramp does not rise.
	jumpMinRise = 2.0
	// jumpWindowTicks is how long after the airborne rising edge the rise may still arrive (about 62 ms). It
	// absorbs entity snapshots that are several ticks apart.
	jumpWindowTicks = 4
	// teleportRise is the largest believable Z change between two samples. Anything larger is a respawn or
	// teleport (the first tick of a demo has placeholder positions) and resets the detector.
	teleportRise = 64.0
	// reloadDebounceTicks suppresses a clip-refill reload shortly after a reload edge, so one reload is reported
	// once. It is longer than the slowest reload (about 3.8 s at 64 Hz = 243 ticks).
	reloadDebounceTicks = 320
	// zoomUnscopeAfterShotTicks: a sniper rifle unscopes by itself after a shot. A zoom drop to 0 this soon after
	// the clip shrank is not a button press and is not reported.
	zoomUnscopeAfterShotTicks = 2
)

// sample is one player's state at one demo tick.
type sample struct {
	Tick     int
	Alive    bool
	Z        float64
	Airborne bool
	Weapon   int    // entity id of the active weapon, 0 if none
	WeapName string // display name of the active weapon, the "weapon" column of derived events
	InReload bool   // active weapon's m_bInReload, false when the demo does not carry the property
	Clip     int    // rounds in the active weapon's magazine, -1 when unknown
	Zoom     int    // active weapon's zoom level (0 none, 1 half, 2 full)
}

// derivedEvent is a reconstructed game event of one player.
type derivedEvent struct {
	Tick   int
	Name   string // player_jump, weapon_reload or weapon_zoom
	Weapon string
}

// detector turns one player's consecutive samples into derived events. The zero value is ready to use.
type detector struct {
	prev    sample
	hasPrev bool
	groundZ float64 // Z of the last grounded sample

	edgeTick int // tick of a take-off that is still waiting for its rise
	edgeSet  bool

	lastReload   int // tick of the last reload event, for the debounce
	lastReloadOK bool
	lastShot     int // tick at which the clip last shrank
	lastShotOK   bool
}

// step feeds the next sample (ticks strictly increasing) and returns the events it completes.
func (d *detector) step(s sample) []derivedEvent {
	var out []derivedEvent
	p, ok := d.prev, d.hasPrev
	d.prev, d.hasPrev = s, true
	if !s.Alive || !ok || !p.Alive || abs(s.Z-p.Z) > teleportRise {
		*d = detector{prev: s, hasPrev: true, groundZ: s.Z}
		return nil
	}

	// player_jump: the airborne rising edge, confirmed by a rise above the last grounded Z.
	switch {
	case !s.Airborne:
		d.groundZ, d.edgeSet = s.Z, false
	case !p.Airborne:
		d.edgeTick, d.edgeSet = s.Tick, true
	}
	if d.edgeSet && s.Airborne {
		switch {
		case s.Z-d.groundZ >= jumpMinRise:
			out = append(out, derivedEvent{Tick: d.edgeTick, Name: "player_jump", Weapon: s.WeapName})
			d.edgeSet = false
		case s.Tick-d.edgeTick >= jumpWindowTicks:
			d.edgeSet = false
		}
	}

	if s.Weapon == 0 || s.Weapon != p.Weapon {
		return out // a weapon switch changes clip, reload flag and zoom without any button press
	}
	// weapon_reload: the reload flag rises, or (when the demo lacks the flag) the magazine refills.
	refill := p.Clip >= 0 && s.Clip > p.Clip
	debounced := d.lastReloadOK && s.Tick-d.lastReload < reloadDebounceTicks
	if (s.InReload && !p.InReload) || (refill && !debounced) {
		out = append(out, derivedEvent{Tick: s.Tick, Name: "weapon_reload", Weapon: s.WeapName})
		d.lastReload, d.lastReloadOK = s.Tick, true
	}
	if s.Clip >= 0 && p.Clip >= 0 && s.Clip < p.Clip {
		d.lastShot, d.lastShotOK = s.Tick, true
	}
	// weapon_zoom: any change of the zoom level, except the automatic unscope after a shot.
	if s.Zoom != p.Zoom {
		auto := s.Zoom == 0 && d.lastShotOK && s.Tick-d.lastShot <= zoomUnscopeAfterShotTicks
		if !auto {
			out = append(out, derivedEvent{Tick: s.Tick, Name: "weapon_zoom", Weapon: s.WeapName})
		}
	}
	return out
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
