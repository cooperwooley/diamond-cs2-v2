package main

import (
	"compress/gzip"
	"encoding/csv"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/msg"
)

// csvFile is a CSV writer over an optional gzip stream.
type csvFile struct {
	f  *os.File
	gz *gzip.Writer
	w  *csv.Writer
}

func createCSV(path string, header []string) (*csvFile, error) {
	return openCSV(path, header, true)
}

func createPlainCSV(path string, header []string) (*csvFile, error) {
	return openCSV(path, header, false)
}

func openCSV(path string, header []string, compress bool) (*csvFile, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	c := &csvFile{f: f}
	if compress {
		c.gz = gzip.NewWriter(f)
		c.w = csv.NewWriter(c.gz)
	} else {
		c.w = csv.NewWriter(f)
	}
	return c, c.w.Write(header)
}

// Write appends a row. Errors surface from Close (csv.Writer buffers).
func (c *csvFile) Write(cells []string) { _ = c.w.Write(cells) }

func (c *csvFile) Close() error {
	c.w.Flush()
	err := c.w.Error()
	if c.gz != nil {
		if e := c.gz.Close(); err == nil {
			err = e
		}
	}
	if e := c.f.Close(); err == nil {
		err = e
	}
	return err
}

// eventTable is the rows of one event name. Columns are the union of the keys seen, in first-seen order.
type eventTable struct {
	cols []string
	seen map[string]bool
	rows []map[string]string
}

func (t *eventTable) add(row map[string]string) {
	for k := range row {
		if !t.seen[k] {
			t.seen[k] = true
			t.cols = append(t.cols, k)
		}
	}
	t.rows = append(t.rows, row)
}

// eventTables holds the raw game events and the derived ones, which are only written for names the demo lacks.
type eventTables struct {
	raw, derived map[string]*eventTable
}

func newEventTables() *eventTables {
	return &eventTables{raw: map[string]*eventTable{}, derived: map[string]*eventTable{}}
}

func tableFor(m map[string]*eventTable, name string) *eventTable {
	t := m[name]
	if t == nil {
		t = &eventTable{seen: map[string]bool{}}
		m[name] = t
	}
	return t
}

// addRaw stores a game event as received. The “userid“ key is renamed user_id; every other key keeps its name.
func (e *eventTables) addRaw(tick int, ev events.GenericGameEvent) {
	row := map[string]string{"tick": strconv.Itoa(tick), "derived": "false"}
	for k, v := range ev.Data {
		if k == "userid" {
			k = "user_id"
		}
		row[k] = eventValue(v)
	}
	// weapon is the raw string ("weapon_ak47"); weapon_name is the display name ("AK-47") that
	// active_weapon_name uses, on raw and derived events alike.
	if w, ok := row["weapon"]; ok {
		if eq := common.MapEquipment(w); eq != common.EqUnknown {
			row["weapon_name"] = eq.String()
		}
	}
	tableFor(e.raw, ev.Name).add(row)
}

func (e *eventTables) addDerived(tick int, name string, userID int, weapon string) {
	tableFor(e.derived, name).add(map[string]string{
		"tick": strconv.Itoa(tick), "derived": "true", "user_id": strconv.Itoa(userID), "weapon_name": weapon,
	})
}

func eventValue(v *msg.CMsgSource1LegacyGameEventKeyT) string {
	switch {
	case v == nil:
		return ""
	case v.ValString != nil:
		return *v.ValString
	case v.ValFloat != nil:
		return strconv.FormatFloat(float64(*v.ValFloat), 'f', -1, 32)
	case v.ValLong != nil:
		return strconv.Itoa(int(*v.ValLong))
	case v.ValShort != nil:
		return strconv.Itoa(int(*v.ValShort))
	case v.ValByte != nil:
		return strconv.Itoa(int(*v.ValByte))
	case v.ValBool != nil:
		return strconv.FormatBool(*v.ValBool)
	case v.ValUint64 != nil:
		return strconv.FormatUint(*v.ValUint64, 10)
	}
	return ""
}

// write puts one events/<name>.csv.gz per event name: the raw table if the demo has the event, else the derived one.
func (e *eventTables) write(dir string) error {
	out := map[string]*eventTable{}
	for name, t := range e.derived {
		out[name] = t
	}
	for name, t := range e.raw {
		out[name] = t
	}
	names := make([]string, 0, len(out))
	for name := range out {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		t := out[name]
		cols := append([]string{"tick", "derived"}, without(t.cols, "tick", "derived")...)
		f, err := createCSV(filepath.Join(dir, name+".csv.gz"), cols)
		if err != nil {
			return err
		}
		cells := make([]string, len(cols))
		for _, row := range t.rows {
			for i, c := range cols {
				cells[i] = row[c]
			}
			f.Write(cells)
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	return nil
}

func without(cols []string, drop ...string) []string {
	var out []string
	for _, c := range cols {
		skip := false
		for _, d := range drop {
			skip = skip || c == d
		}
		if !skip {
			out = append(out, c)
		}
	}
	return out
}
