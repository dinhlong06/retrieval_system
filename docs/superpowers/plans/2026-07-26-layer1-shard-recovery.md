# Layer 1 Shard Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover shot detection for `K20_V029`, merge complete Layer 1 outputs for all 605 videos, and prevent stale claims from blocking later coordinator runs.

**Architecture:** Keep `.done` files as durable completion state and treat claim files as disposable state for one coordinator run. At startup, the coordinator clears only regular files in its stage claim directory and immediately rebuilds them from all shard `.done` files before launching workers.

**Tech Stack:** Bash, Docker, TransNetV2, ffmpeg, JSONL, NFSv4.

## Global Constraints

- Change only `layer_1/run_shards.sh`; create no runtime files, classes, configuration flags, or compatibility paths.
- Do not change TransNetV2, ffmpeg selection, ASR, schemas, or downstream layers.
- Do not use `FORCE=1` for merge.
- Assume one coordinator per stage; stop the old worker before rebuilding claims.
- Follow YAGNI, KISS, and minimalist rules from `CLAUDE.md`.
- The workspace root is not a Git repository, so commit steps are unavailable.

---

## File Structure

- Modify: `layer_1/run_shards.sh` — reset stage-local claim files and reseed them from durable `.done` markers before starting workers.
- Read only: `layer_1/gpu_shot_and_asr.py` — existing worker claim and completion semantics.
- Read only: `layer_1/shards/**` — recovery inputs and verification evidence.
- No persistent test file: use an isolated `/tmp` copy of the current coordinator and marker state for the regression cycle.

### Task 1: Reset Stale Claims at Coordinator Startup

**Files:**
- Modify: `layer_1/run_shards.sh:49-55`
- Test: isolated temporary copy created from `layer_1/run_shards.sh` and `layer_1/shards/shots_*/shots.jsonl.done`

**Interfaces:**
- Consumes: `layer_1/shards/shots_*/shots.jsonl.done` files whose non-empty lines are completed `video_id` values.
- Produces: one regular file named by completed `video_id` inside `layer_1/shards/claims_shots`, before any worker starts.

- [ ] **Step 1: Run an isolated failing regression check**

Run from the project root:

```bash
probe_root=$(mktemp -d /tmp/ai26-claims-test.XXXXXX)
mkdir -p "$probe_root/layer_1/shards" "$probe_root/dataset"
cp layer_1/run_shards.sh "$probe_root/layer_1/"
cp -a layer_1/shards/claims_shots "$probe_root/layer_1/shards/"
for src in layer_1/shards/shots_*; do
  dst="$probe_root/layer_1/shards/$(basename "$src")"
  mkdir -p "$dst"
  cp "$src/shots.jsonl.done" "$dst/"
done
touch "$probe_root/layer_1/shards/claims_shots/STALE_TEST_VIDEO"
ln -s "$(readlink -f dataset/video)" "$probe_root/dataset/video"
expected_done=$(cat "$probe_root"/layer_1/shards/shots_*/shots.jsonl.done | sort -u | wc -l)
NSHARDS=0 "$probe_root/layer_1/run_shards.sh" shots
test "$(find "$probe_root/layer_1/shards/claims_shots" -maxdepth 1 -type f | wc -l)" -eq "$expected_done"
test ! -e "$probe_root/layer_1/shards/claims_shots/STALE_TEST_VIDEO"
```

Expected before the fix: the count assertion exits non-zero and `STALE_TEST_VIDEO` remains because old claims are never rebuilt.

- [ ] **Step 2: Implement the minimal coordinator reset**

Apply this change immediately after the claim directory is created and before existing `.done` seeding:

```bash
mkdir -p "$CLAIMS"
# Claim chỉ có hiệu lực trong một lần chạy; .done mới là trạng thái hoàn thành bền vững.
find "$CLAIMS" -mindepth 1 -maxdepth 1 -type f -delete
```

Do not add worker-side cleanup, retries, leases, or new flags.

- [ ] **Step 3: Run syntax validation**

Run:

```bash
bash -n layer_1/run_shards.sh
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Re-run the isolated regression check**

Repeat Step 1 with a new `probe_root`.

Expected after the fix: the first status line ends with `đã gieo $expected_done claim`, followed by:

```text
== shots xong. Chạy './run_shards.sh merge' để gộp. ==
```

Both assertions exit 0: the claim count equals the durable completion count and `STALE_TEST_VIDEO` is absent.

- [ ] **Step 5: Record the checkpoint without committing**

Run:

```bash
git rev-parse --show-toplevel
```

Expected: `fatal: not a git repository`; retain the verified workspace edit without creating a commit.

### Task 2: Recover `K20_V029`

**Files:**
- Modify through existing pipeline output: `layer_1/shards/shots_0/shots.jsonl`
- Modify through existing pipeline output: `layer_1/shards/shots_0/shots.jsonl.done`
- Read only: `dataset/video/K20_V029.mp4`

**Interfaces:**
- Consumes: one readable MP4 and the repaired coordinator claim state.
- Produces: shot JSONL entries and one durable `.done` marker for `K20_V029`.

- [ ] **Step 1: Resolve the exact old worker before stopping it**

Run from `layer_1`:

```bash
docker ps --filter ancestor=ai26-layer1 --format '{{.ID}} {{.Status}} {{.Command}}'
matching_id=""
for id in $(docker ps --filter ancestor=ai26-layer1 --format '{{.ID}}'); do
  output=$(docker inspect "$id" --format '{{range .Mounts}}{{if eq .Destination "/data/output"}}{{.Source}}{{end}}{{end}}')
  cmd=$(docker inspect "$id" --format '{{json .Config.Cmd}}')
  if [[ "$output" == "$PWD/shards/shots_1" && "$cmd" == *--skip_asr* ]]; then
    [[ -z "$matching_id" ]] || { echo "Tìm thấy nhiều worker shots_1" >&2; exit 1; }
    matching_id="$id"
  fi
done
printf 'matching_id=%s\n' "$matching_id"
```

Expected: at most one running container whose `/data/output` source is `layer_1/shards/shots_1` and whose command includes `--skip_asr`. Do not stop any container with a different output mount.

- [ ] **Step 2: Stop only the resolved stale worker**

Run from `layer_1` in the same shell used by Step 1:

```bash
if [[ -n "$matching_id" ]]; then
  docker stop --time 10 "$matching_id"
fi
```

Expected: Docker prints that same container ID. If no matching container remains, continue without stopping anything.

- [ ] **Step 3: Verify the complete source file is readable**

Run from `layer_1`:

```bash
timeout 60 dd if=../dataset/video/K20_V029.mp4 of=/dev/null bs=4M status=none
```

Expected: exit code 0 within 60 seconds. If it times out or fails, stop this task and leave all 604 completed shard outputs unchanged.

- [ ] **Step 4: Retry only the missing video with one shard**

Run from `layer_1`:

```bash
NSHARDS=1 ./run_shards.sh shots --videos K20_V029.mp4
```

Expected: the worker processes only `K20_V029`, appends its shots to `shards/shots_0/shots.jsonl`, appends `K20_V029` to `shards/shots_0/shots.jsonl.done`, and exits successfully.

- [ ] **Step 5: Verify the recovered video before merge**

Run from the project root:

```bash
rg -l '^K20_V029$' layer_1/shards/shots_*/shots.jsonl.done
rg -l '"video_id": "K20_V029"' layer_1/shards/shots_*/shots.jsonl
cat layer_1/shards/shots_*/shots.jsonl.done | sort | uniq -d | rg '^K20_V029$' && exit 1 || true
```

Expected: exactly one `.done` file and its corresponding JSONL file are printed; the duplicate-marker check prints nothing.

### Task 3: Merge and Verify Complete Layer 1 Outputs

**Files:**
- Modify through existing merge: `layer_1/shots.jsonl`
- Modify through existing merge: `layer_1/shots.jsonl.done`
- Rebuild unchanged ASR merge: `layer_1/whisper.jsonl`
- Rebuild unchanged ASR completion markers: `layer_1/whisper.jsonl.done`

**Interfaces:**
- Consumes: all per-shard shot and ASR JSONL files plus `.done` markers.
- Produces: complete merged Layer 1 JSONL files for downstream processing.

- [ ] **Step 1: Run strict merge**

Run from `layer_1`:

```bash
./run_shards.sh merge
```

Expected: both lines report `605/605`; no command uses `FORCE=1`.

- [ ] **Step 2: Validate JSON syntax for every shard and merged output**

Run from the project root:

```bash
for f in layer_1/shards/shots_*/shots.jsonl layer_1/shards/asr_*/whisper.jsonl layer_1/shots.jsonl layer_1/whisper.jsonl; do
  python3 -m json.tool --json-lines "$f" >/dev/null
done
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Validate complete video coverage**

Run from the project root:

```bash
comm -23 <((cd dataset/video && printf '%s\n' *.mp4) | sed 's/\.mp4$//' | sort -u) <(sed -n 's/.*"video_id": "\([^"]*\)".*/\1/p' layer_1/shots.jsonl | sort -u)
sed -n 's/.*"video_id": "\([^"]*\)".*/\1/p' layer_1/shots.jsonl | sort -u | wc -l
```

Expected: the set-difference command prints nothing and the count is `605`.

- [ ] **Step 4: Validate shot identity uniqueness**

Run from the project root:

```bash
sed -n 's/.*"shot_id": "\([^"]*\)".*/\1/p' layer_1/shots.jsonl | sort | uniq -d | sed -n '1,20p'
sed -n 's/.*"shot_id": "\([^"]*\)".*/\1/p' layer_1/shots.jsonl | sort | uniq -d | wc -l
```

Expected: the first command prints nothing and the count is `0`.

- [ ] **Step 5: Confirm downstream preconditions without running downstream work**

Run from the project root:

```bash
wc -l layer_1/shots.jsonl.done layer_1/whisper.jsonl.done
comm -3 <(sort -u layer_1/shots.jsonl.done) <(sort -u layer_1/whisper.jsonl.done)
```

Expected: both marker files contain 605 lines and the set comparison prints nothing.
