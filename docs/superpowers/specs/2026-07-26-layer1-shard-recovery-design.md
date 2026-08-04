# Layer 1 Shard Recovery Design

## Status

Approved in principle on 2026-07-26. This document defines the exact implementation scope for final review.

## Context

Layer 1 has complete ASR output for all 605 videos, but shot detection has completed only 604. The missing video is `K20_V029`. Its worker claimed the video and then stalled while ffmpeg was reading the source from an unstable, full NFS mount.

Claims are currently empty files created before processing. A claim remains after a worker fails or is stopped, while completion is recorded separately in `shots.jsonl.done`. On a later run, the stale claim prevents every shard from retrying that unfinished video.

The merge guard is behaving correctly and must remain strict. Forcing a partial merge would propagate the missing video into shot-transcript mapping, keyframe extraction, captioning, embeddings, OCR, object detection, and database indexing.

## Goals

- Recover shot detection for `K20_V029` and produce a complete `605/605` merge.
- Make `.done` the source of truth when a new shard coordinator starts.
- Prevent claims left by an earlier stopped or failed run from blocking future retries.
- Keep the change local to `layer_1/run_shards.sh`.

## Non-goals

- Building a generic retry scheduler, claim lease system, watchdog, or NFS monitor.
- Adding new configuration flags, files, classes, or compatibility paths.
- Changing TransNetV2, ffmpeg selection, ASR, output schemas, or downstream layers.
- Making concurrent coordinator invocations safe; the current workflow assumes one coordinator per stage.

## Design

At coordinator startup, before workers are launched:

1. Ensure the stage claim directory exists.
2. Delete only regular claim files immediately inside that directory.
3. Recreate claims from the union of all shard `.done` files for the selected stage.
4. Launch workers using the existing atomic `O_CREAT | O_EXCL` claim operation.

This makes stale claims disposable coordination state and keeps `.done` as durable completion state. Clearing claims is safe only before workers start; the implementation will remain in the coordinator startup path immediately before seeding and launching workers.

No cleanup is added inside individual workers. That would allow another live shard to retry a failing video during the same run and adds coordination behavior that is not needed for the current failure.

## Recovery Procedure

Run the following procedure from the `layer_1` directory.

1. Confirm the existing `shots` worker is still stalled and stop it before rebuilding claims.
2. Require a complete read of `K20_V029.mp4` to finish successfully before retrying:

   ```bash
   timeout 60 dd if=../dataset/video/K20_V029.mp4 of=/dev/null bs=4M status=none
   ```
3. Apply the coordinator claim-reset change.
4. Run one shard for only the missing video:

   ```bash
   NSHARDS=1 ./run_shards.sh shots --videos K20_V029.mp4
   ```

5. Run the existing merge command without `FORCE=1`.
6. Start downstream layers only after the merge reports `605/605` videos.

## Failure Handling

- If the complete-read command times out or fails, do not retry repeatedly and do not force the merge. Leave the 604 completed shard outputs unchanged.
- If the retry stops again, its claim may remain during that run, but the next coordinator invocation will discard it because it is absent from `.done`.
- If merge still reports fewer than 605 videos, use the set difference between dataset video IDs and all shard `.done` files to identify the remaining video; do not infer completion from JSONL rows.

## Verification

- `bash -n layer_1/run_shards.sh` succeeds.
- Before retry, `K20_V029` is present in claims but absent from every shot `.done` file.
- On a fresh coordinator run, completed-video claims are recreated and the stale `K20_V029` claim is removed before a worker claims it again.
- After retry, `K20_V029` has shot rows and appears exactly once across shot `.done` files.
- All shard JSONL files parse successfully.
- Merged `shots.jsonl` contains 605 unique `video_id` values and no duplicate `shot_id` values.
- `./run_shards.sh merge` completes without `FORCE=1` and reports `605/605` for both shots and whisper.

## Scope Check

The design changes one shell script and one proven failure mode. It introduces no speculative retry system or new abstraction, matching the repository's YAGNI, KISS, and minimalist rules.
