# PC setup

Written from the #2 paste-back (2026-11-03). The desktop runs every PC-side stage: development, tests and eval in WSL,
and the stages that drive CS2/HLAE (`generate`, `render`, `supervise`) plus `play` under Windows Python.

## Layout

| What | Where |
|---|---|
| WSL dev clone | `~/mst/cs5404/diamond-cs2-v2`, conda env `dcs2` (Python 3.10, torch 2.14.1+cu130) |
| Windows runtime clone | `C:\dcs2\diamond-cs2-v2` (update with `git pull` only; nothing auto-pulls) |
| Windows venv | `C:\dcs2\venv` (Python 3.10.11, torch 2.14.1+cu126, demoparser2 0.42.1) |
| Data root (`DCS2_ROOT`) | `C:\dcs2\data`, seen from WSL as `/mnt/c/dcs2/data` |
| HLAE (`DCS2_HLAE_DIR`) | `C:\hlae`, version 2.192.10 (AfxHookSource2 0.41.9) |
| CS2 (`DCS2_CS2_DIR`) | `D:\Steam\steamapps\common\Counter-Strike Global Offensive` (Steam lives on `D:`, not the default path) |
| GPU | RTX 3060 12 GB, driver 610.88 |

`DCS2_NTFY_TOPIC` holds the phone-notification topic. It is a user environment variable and is never committed.
`DCS2_MILL_HOST` (`<sso>@mill.mst.edu`) is set per shell for `cs2wm sync` (see below).

## Recorded values

- **CS2 build:** `buildid` 26393695 (`D:\Steam\steamapps\appmanifest_730.acf`).
- **Updates:** CS2 stays on Steam's default auto-update, by the owner's choice. A patch is caught by the supervisor's
  patch guard: when the `buildid` changes, a 10 s render smoke test runs before any generation, and generation stops with
  an alert if it fails.
- **Sensitivity:** 1.15, so `play.sensitivity` is 1.15. Labels are degree-based, so the action bins do not depend on it.
- **Video settings** (`cs2_video.txt`):
  - CPU, GPU and GPU-memory levels are 3; MSAA 8×; shadows 2 with dynamic shadows on; texture, particle and AO detail 0;
    FSR off; shader quality 1; texture filtering 0; vsync off.
  - Boost player contrast (`r_player_visibility_mode`) is on.
  - The owner plays fullscreen at 1280×960 (4:3). The pipeline launches with `-sw -w 1280 -h 720`, which overrides the
    resolution for that session.
  - CS2 can save the override when it exits, so the owner's file is backed up at
    `C:\dcs2\data\backup\cs2_video.user.txt` (its original path is in `cs2_video.path.txt` next to it). Restore it after
    a pipeline session if CS2 next opens windowed.

## Reproduce

WSL:
```bash
conda activate dcs2
pip install -r requirements.txt -e ".[dev]"
pip install --force-reinstall --no-deps "torch==2.14.1" "torchvision==0.29.1"   # Linux PyPI wheels include CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Windows (PowerShell):
```powershell
winget install -e --id Python.Python.3.10 --scope user
git clone https://github.com/cooperwooley/diamond-cs2-v2.git C:\dcs2\diamond-cs2-v2
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv C:\dcs2\venv
C:\dcs2\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
C:\dcs2\venv\Scripts\pip install -r C:\dcs2\diamond-cs2-v2\requirements.txt
C:\dcs2\venv\Scripts\pip install -e C:\dcs2\diamond-cs2-v2
```

HLAE comes from the release zip at https://github.com/advancedfx/advancedfx/releases, extracted to `C:\hlae` and
unblocked (`Get-ChildItem -Recurse C:\hlae | Unblock-File`).

## Demo exporter (`demodump`)

Demos are parsed by `demodump`, a Go program built from `tools/demodump` (#72), not by demoparser2. Build it in WSL
(needs Go; the PC never needs it):
```bash
scripts/build_demodump.sh        # writes bin/demodump (Linux) and bin/demodump.exe (Windows), both git-ignored
cp bin/demodump.exe /mnt/c/dcs2/bin/demodump.exe
```
and point Windows at it: `setx DCS2_DEMODUMP C:\dcs2\bin\demodump.exe`. The lookup order is env `DCS2_DEMODUMP`, then
`demo.demodump` in `cs2wm.yaml`, then `bin\demodump.exe` in the repo, then `PATH`. A missing binary fails with the
build script's name. Each demo is parsed once and the tables are cached in `<DCS2_ROOT>\demo_cache` (keyed by path, size
and mtime; safe to delete). `demo.backend: demoparser2` switches back to the old parser.

## Windows Smart App Control blocks torchvision

On this PC, Windows Smart App Control (WinError 4551, "An Application Control policy has blocked this file") blocks
torchvision's compiled extension `torchvision\_C_stable.pyd`, so `import torchvision` fails in `C:\dcs2\venv` with
`operator torchvision::nms does not exist`. torch itself, h5py, OpenCV, demoparser2 and `demodump.exe` load fine.

- The Windows pipeline stages don't import torchvision: `pack` resizes with `torch.nn.functional.interpolate`, which is
  what v1's torchvision resize does for float tensors (bit-identical; `tests/test_pack.py` checks it against
  torchvision on Linux).
- Anything that needs torchvision, such as the LPIPS evals (#27, #33, via the `lpips` package), runs in WSL or on the
  Mill instead.
- Don't turn Smart App Control off to work around this: it can't be turned back on without reinstalling Windows.

## Syncing to the Mill without a key

There is no SSH key for the Mill, so nothing syncs unattended. `sync.auto` and `sync.batch_mode` are both `false`
(#63): nightly runs stop each demo at `packed`, and the upload is run by hand, prompting for the password once:
```powershell
C:\dcs2\venv\Scripts\Activate.ps1
$env:DCS2_MILL_HOST = "<sso>@mill.mst.edu"
cs2wm sync; cs2wm status
```
