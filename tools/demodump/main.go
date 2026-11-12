// Command demodump parses a CS2 demo with demoinfocs-golang and writes it as tables that cs2wm/demo.py loads.
//
// Usage: demodump <demo.dem> <outdir>, or demodump --version.
//
// Files written to outdir (gzip-compressed CSV, one header row; see cs2wm/demo.py for how they are read):
//
//	header.json       map, patch version, tick rate, first and last tick, tool versions
//	players.csv       one row per player and user_id occupancy: user_id, entity_id, name, steamid, ..., first/last tick
//	ticks.csv.gz      one row per (tick, player); ticks are demo ticks (IngameTick), de-duplicated
//	events/<name>.csv.gz  every game event, one table per event name, plus the derived events (derived=true)
//	grenades.csv.gz   grenade trajectories, one row per (grenade, tick)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"strconv"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/msg"
)

// version is the demodump output format version. Bump it when a column or its meaning changes: it is part of the
// cache key on the Python side.
const version = "1"

// behaviorVersion identifies what the tables contain for a given demo: the derived-event rules and constants in
// derive.go and the table logic. Bump it whenever either changes; the Python cache key includes it. The pinned hash
// in behavior_test.go fails until you do.
const behaviorVersion = "1"

const (
	demoinfocsModule = "github.com/markus-wa/demoinfocs-golang/v6"
	// duckProp is the pawn property that holds the crouch amount (0 standing .. 1 fully ducked), the same value
	// demoparser2 calls duck_amount.
	duckProp = "m_pMovementServices.m_flDuckAmount"
	// reloadProp is the weapon property that is true while a reload is in progress.
	reloadProp = "m_bInReload"
)

var tickColumns = []string{
	"tick", "user_id", "entity_id", "pawn_entity_id", "name", "steamid", "team", "is_alive", "health",
	"X", "Y", "Z", "yaw", "pitch", "is_airborne", "duck_amount", "active_weapon_name", "is_walking", "is_scoped",
}

var grenadeColumns = []string{
	"grenade_entity_id", "tick", "grenade_type", "thrower_user_id", "thrower_steamid", "X", "Y", "Z",
}

// demoinfocsVersion is the demoinfocs-golang version, plus "+<version>" of a fork that replaces it in go.mod, so a
// patched build never passes for the release (the Python cache key includes it).
func demoinfocsVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			if dep.Path == demoinfocsModule {
				if dep.Replace != nil {
					return dep.Version + "+" + dep.Replace.Version
				}
				return dep.Version
			}
		}
	}
	return "unknown"
}

func main() {
	showVersion := flag.Bool("version", false, "print the demodump and demoinfocs versions and exit")
	flag.Parse()
	if *showVersion {
		fmt.Printf("demodump %s demoinfocs %s behavior %s\n", version, demoinfocsVersion(), behaviorVersion)
		return
	}
	if flag.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: demodump <demo.dem> <outdir>")
		os.Exit(2)
	}
	if err := dump(flag.Arg(0), flag.Arg(1)); err != nil {
		fmt.Fprintln(os.Stderr, "demodump:", err)
		os.Exit(1)
	}
}

// occupancy is one player holding a user_id for a span of ticks. A user_id that is reused by someone else gets a
// second occupancy, so events can be joined to the right name by tick.
type occupancy struct {
	userID, entityID, pawnID, team, first, last int
	name                                        string
	steamID                                     uint64
	bot                                         bool
}

// frameRow is one player's row of the current tick, kept until the tick is known to be final.
type frameRow struct {
	userID int
	cells  []string
	s      sample
}

type dumper struct {
	ticks, grenades *csvFile
	events          *eventTables
	detectors       map[int]*detector
	roster          []*occupancy
	current         map[int]*occupancy

	pending     []frameRow
	pendingTick int
	hasPending  bool
	firstTick   int
	lastTick    int
	rows        int
	seenTick    bool

	grenadeIDs map[*common.GrenadeProjectile]int
	nextGrenID int

	header map[string]any
}

func dump(demoPath, outDir string) error {
	f, err := os.Open(demoPath)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := os.MkdirAll(filepath.Join(outDir, "events"), 0o755); err != nil {
		return err
	}
	d := &dumper{
		events:     newEventTables(),
		detectors:  map[int]*detector{},
		current:    map[int]*occupancy{},
		grenadeIDs: map[*common.GrenadeProjectile]int{},
		header:     map[string]any{},
	}
	if d.ticks, err = createCSV(filepath.Join(outDir, "ticks.csv.gz"), tickColumns); err != nil {
		return err
	}
	if d.grenades, err = createCSV(filepath.Join(outDir, "grenades.csv.gz"), grenadeColumns); err != nil {
		return err
	}

	p := demoinfocs.NewParser(f)
	defer p.Close()
	p.RegisterNetMessageHandler(func(m *msg.CDemoFileHeader) {
		d.header["map_name"] = m.GetMapName()
		d.header["patch_version"] = m.GetPatchVersion()
		d.header["client_name"] = m.GetClientName()
		d.header["server_name"] = m.GetServerName()
	})
	p.RegisterEventHandler(func(e events.GenericGameEvent) { d.events.addRaw(p.GameState().IngameTick(), e) })
	p.RegisterEventHandler(func(e events.GrenadeProjectileThrow) {
		d.grenadeIDs[e.Projectile] = d.nextGrenID
		d.nextGrenID++
		if th := e.Projectile.Thrower; th != nil {
			d.events.addDerived(p.GameState().IngameTick(), "grenade_thrown", th.UserID, e.Projectile.WeaponInstance.String())
		}
	})
	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) { d.writeGrenade(e.Projectile) })
	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		d.frame(gs.IngameTick(), gs.Participants().Playing())
	})

	// A demo cut short (CS2 killed mid-recording) is still a usable partial dump: everything up to the cut is kept
	// and header.json says truncated. Any other error is a failure and nothing is kept.
	parseErr := p.ParseToEnd(context.Background())
	truncated := errors.Is(parseErr, demoinfocs.ErrUnexpectedEndOfDemo)
	if parseErr != nil && !truncated {
		return fmt.Errorf("parse %s: %w", demoPath, errors.Join(parseErr, d.close()))
	}
	d.header["truncated"] = truncated
	if truncated {
		fmt.Fprintf(os.Stderr, "demodump: %s ends unexpectedly, writing a partial dump\n", demoPath)
	}
	d.flush()
	for _, gp := range p.GameState().GrenadeProjectiles() {
		d.writeGrenade(gp)
	}
	d.header["tick_rate"] = p.TickRate()
	if d.rows == 0 {
		return fmt.Errorf("%s has no player ticks", demoPath)
	}
	if err := d.events.write(filepath.Join(outDir, "events")); err != nil {
		return err
	}
	if err := d.writePlayers(filepath.Join(outDir, "players.csv")); err != nil {
		return err
	}
	if err := d.close(); err != nil {
		return err
	}
	return d.writeHeader(filepath.Join(outDir, "header.json"), demoPath)
}

func (d *dumper) close() error {
	return errors.Join(d.ticks.Close(), d.grenades.Close())
}

// frame records the state of every playing player at one demo tick. Client demos hold a few repeated ticks (the
// last frame of a tick wins) and, rarely, a tick that goes backwards (skipped).
func (d *dumper) frame(tick int, playing []*common.Player) {
	if d.hasPending && tick < d.pendingTick {
		return
	}
	rows := make([]frameRow, 0, len(playing))
	for _, pl := range playing {
		if pl != nil {
			rows = append(rows, d.row(tick, pl))
		}
	}
	if d.hasPending && tick > d.pendingTick {
		d.flush()
	}
	d.pending, d.pendingTick, d.hasPending = rows, tick, true
}

// row snapshots one player.
func (d *dumper) row(tick int, pl *common.Player) frameRow {
	pos := pl.Position()
	pawn := pl.PlayerPawnEntity()
	st := playerState{
		Tick: tick, UserID: pl.UserID, EntityID: pl.EntityID, Team: int(pl.Team), Health: pl.Health(),
		Name: pl.Name, SteamID: pl.SteamID64, HasPawn: pawn != nil, Alive: pl.IsAlive(),
		X: pos.X, Y: pos.Y, Z: pos.Z, Yaw: float64(pl.ViewDirectionX()), Pitch: float64(pl.ViewDirectionY()),
		Airborne: pl.IsAirborne(), Walking: pl.IsWalking(), Scoped: pl.IsScoped(),
	}
	if pl.Entity != nil {
		st.EntityID = pl.Entity.ID()
	}
	s := sample{Tick: tick, Z: pos.Z, Airborne: st.Airborne, Clip: -1}
	if pawn != nil {
		st.PawnID = pawn.ID()
		if v, ok := pawn.PropertyValue(duckProp); ok && v.Any != nil {
			st.Duck = float64(v.Float())
		}
	}
	if w := pl.ActiveWeapon(); w != nil {
		st.Weapon = w.String()
		s.WeapName, s.Clip, s.Zoom = st.Weapon, w.AmmoInMagazine(), int(w.ZoomLevel())
		if w.Entity != nil {
			s.Weapon = w.Entity.ID()
			if v, ok := w.Entity.PropertyValue(reloadProp); ok && v.Any != nil {
				s.InReload = v.BoolVal()
			}
		}
	}
	s.Alive = st.alive() && st.validPose() // a row without a pose resets the derived-event detectors
	d.occupy(st, pl.IsBot)
	return frameRow{userID: pl.UserID, cells: st.cells(), s: s}
}

// occupy records who holds the user_id at this tick, opening a new occupancy when the player changed.
func (d *dumper) occupy(st playerState, bot bool) {
	cur := d.current[st.UserID]
	if cur == nil || cur.name != st.Name || cur.steamID != st.SteamID || cur.entityID != st.EntityID {
		cur = &occupancy{userID: st.UserID, first: st.Tick, bot: bot}
		d.current[st.UserID] = cur
		d.roster = append(d.roster, cur)
	}
	cur.entityID, cur.pawnID, cur.team, cur.last = st.EntityID, st.PawnID, st.Team, st.Tick
	cur.name, cur.steamID = st.Name, st.SteamID
}

// flush writes the pending tick and runs the derived-event detectors on it.
func (d *dumper) flush() {
	if !d.hasPending {
		return
	}
	for _, r := range d.pending {
		d.ticks.Write(r.cells)
		d.rows++
		det := d.detectors[r.userID]
		if det == nil {
			det = &detector{}
			d.detectors[r.userID] = det
		}
		for _, e := range det.step(r.s) {
			d.events.addDerived(e.Tick, e.Name, r.userID, e.Weapon)
		}
	}
	if !d.seenTick {
		d.firstTick, d.seenTick = d.pendingTick, true
	}
	d.lastTick = d.pendingTick
	d.pending, d.hasPending = nil, false
}

func (d *dumper) writeGrenade(gp *common.GrenadeProjectile) {
	id, ok := d.grenadeIDs[gp]
	if !ok {
		id = d.nextGrenID // a grenade already in flight when the demo starts
		d.nextGrenID++
		d.grenadeIDs[gp] = id
	}
	uid, sid, kind := -1, uint64(0), ""
	if gp.Thrower != nil {
		uid, sid = gp.Thrower.UserID, gp.Thrower.SteamID64
	}
	if gp.WeaponInstance != nil {
		kind = gp.WeaponInstance.String()
	}
	for _, t := range gp.Trajectory {
		d.grenades.Write([]string{
			strconv.Itoa(id), strconv.Itoa(t.Tick), kind, strconv.Itoa(uid), strconv.FormatUint(sid, 10),
			fmtFloat(t.Position.X, 3), fmtFloat(t.Position.Y, 3), fmtFloat(t.Position.Z, 3),
		})
	}
	delete(d.grenadeIDs, gp)
}

func (d *dumper) writePlayers(path string) error {
	out, err := createPlainCSV(path, []string{
		"user_id", "entity_id", "pawn_entity_id", "name", "steamid", "is_bot", "team", "first_tick", "last_tick",
	})
	if err != nil {
		return err
	}
	for _, p := range d.roster {
		out.Write([]string{
			strconv.Itoa(p.userID), strconv.Itoa(p.entityID), strconv.Itoa(p.pawnID), p.name,
			strconv.FormatUint(p.steamID, 10), strconv.FormatBool(p.bot), strconv.Itoa(p.team),
			strconv.Itoa(p.first), strconv.Itoa(p.last),
		})
	}
	return out.Close()
}

func (d *dumper) writeHeader(path, demoPath string) error {
	d.header["first_tick"] = d.firstTick
	d.header["last_tick"] = d.lastTick
	d.header["n_players"] = len(d.current)
	d.header["demo_file"] = filepath.Base(demoPath)
	d.header["demodump_version"] = version
	d.header["demoinfocs_version"] = demoinfocsVersion()
	b, err := json.MarshalIndent(d.header, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func fmtFloat(v float64, prec int) string { return strconv.FormatFloat(v, 'f', prec, 64) }
