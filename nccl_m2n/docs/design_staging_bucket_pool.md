# Bounded staging-buffer buckets

Status: implemented. The main implementation is
[`reshard_transpose.cc`](../src/reshard_transpose.cc); configuration is parsed
in [`m2n_config.cc`](../src/m2n_config.cc), and PACKWINDOW host-RMA ordering is
implemented in [`reshard_user_window.cu`](../src/reshard_user_window.cu).

## Scope

`NCCL_RESHARD_STAGING_BUCKETS` replaces PACKWINDOW's fixed per-communicator
transpose allocation with configured size classes:

```text
NCCL_RESHARD_STAGING_BUCKETS="268435456:4,1853358080:2"
```

Each item is `bytes:slots`. The variable is opt-in; when it is unset, the
existing per-communicator allocation remains unchanged. Bucketed staging does
not control the separate channelized staging pipeline in `staging_buffer.cc`.

## Assignment contract

For each CUDA device, a request selects the smallest bucket large enough for
the transfer. Its mapping remains stable for the runtime epoch:

```text
(CUDA device, parent communicator, bucket index) -> physical slot
```

New communicator mappings select healthy physical slots round-robin. More
communicators than slots are allowed, so slots bound device memory and
asynchronous staging lanes rather than communicator identities. A mapping is
not moved if its slot is later poisoned because its cached symmetric-window
registration relies on the stable address.

The physical slot number need not match on every rank. Window registration is
collective on the parent communicator, while split resources use the
configuration-derived bucket index as their rank-stable partition.

## Local and remote ordering

Each slot has a completion event recorded after unpack. A later local user on a
different stream waits for that event before touching the slot.

A local CUDA event does not tell a remote source when a destination slot is
safe to overwrite. The same-NVL host-RMA path therefore uses a per-call lease:

```text
previous destination user completes unpack
    -> destination stream passes the slot completion event
    -> destination sends GRANT to current sources
    -> sources finish packing and wait for all GRANTs
    -> sources issue HPUT + ARRIVAL
    -> destination waits for ARRIVAL and unpacks
    -> destination records slot completion
```

Host-RMA warmup state belongs to the communicator rather than the physical
slot. Sharing a slot therefore cannot make a communicator inherit another
communicator's lazy NCCL host-RMA initialization state.

All deterministic destination plans, offsets, byte counts, and capacity
checks run before protocol entry. Once a GRANT/ARRIVAL operation is attempted,
an error is fail-stop: all ranks must stop M2N submission on that communicator
and coordinate communicator or process-group shutdown. Local resource
quarantine prevents unsafe teardown but is not distributed recovery.

## Host-submission contract

Bucketed staging supports serial host submission with asynchronous CUDA work:

- one reshard host call at a time per process, including across communicators
  that may share a lane;
- overlapping communicator memberships use one consistent logical submission
  order across their ranks;
- calls return after enqueueing, so work on different physical lanes can still
  overlap on the GPU.

Simultaneous host-thread admission is not supported. A rank-local reservation
cannot arbitrate unrelated communicators safely if different communicators win
on different ranks. Opposite orders can also form a distributed dependency
cycle, for example rank X submitting A then B while rank Y submits B then A.

## Failure and lifetime behavior

- New mappings skip poisoned lanes.
- Existing mappings to a poisoned lane fail rather than move to another
  address.
- If a completion event cannot fence outstanding work, the runtime quarantines
  and retains potentially referenced windows, DevComms, split communicators,
  streams, and staging buffers during teardown.
- Communicator mappings and host-RMA warmup records are cleared when the last
  M2N runtime handle is finalized.

## Validation

`staging_slot_reuse_test.cc` covers stable round-robin mapping, local
cross-stream ordering, shared-lane overlap rejection, communicator-owned warmup
state, and healthy-lane selection after poisoning.

`basic_api_test_mpi.cc` adds two focused multi-rank cases. Two warmed
communicators share one destination GPU and one physical slot; one case delays
the previous destination consumer, and the other submits repeated A/B waves
without transfer-phase barriers. Cluster validation must also verify the
`packwindow-lsa-hput` activation log so kernel fallback cannot accidentally
certify the host-RMA lease.
