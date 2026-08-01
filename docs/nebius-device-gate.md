# Nebius CPU/GPU decision run

This is a throughput gate, not a forward-pass microbenchmark. It measures
self-play search, optimizer work, and the three-ratio held-out evaluation, then
normalizes them to one standard ladder iteration. CUDA is accepted only when it
is at least 3x faster end to end and cheaper per equal-work iteration for the
representative candidates.

## Suitable Nebius machines

In `eu-north1`:

| purpose | platform | preset | approximate on-demand price |
|---|---|---|---:|
| GPU gate | `gpu-l40s-a` | `1gpu-8vcpu-32gb` | $1.5484/hour |
| GPU gate, preemptible | same | same | $0.7492/hour |
| 96-worker CPU pool | `cpu-d3` | `96vcpu-384gb` | $2.3808/hour |
| 128-worker CPU pool | `cpu-d3` | `128vcpu-512gb` | $3.1744/hour |

The totals are computed from Nebius's component prices. Check the console quote
before creation because capacity and prices can change. One L40S is enough: the
models are below 200k parameters, so H100/H200 capacity is not the limiting
factor.

## Account-neutral Nebius setup

The examples below deliberately contain no tenant credentials, project IDs,
registry IDs, IP addresses, or SSH keys. Install and authenticate the Nebius CLI,
then supply resources from your own account:

```bash
export PROJECT_ID=<your-project-id>
export REGISTRY_ID=<your-registry-id>
export REGISTRY_HOST=cr.eu-north1.nebius.cloud
export SUBNET_ID=<your-subnet-id>
export CLOUD_INIT_FILE=/absolute/path/to/your/cloud-init.yaml
```

`CLOUD_INIT_FILE` should create your SSH user and contain that developer's public
key. Never commit the rendered file. Create a dedicated service account and give
it read-only access to this registry:

```bash
export BENCHMARK_SA_ID=$(nebius iam service-account create \
  --parent-id "$PROJECT_ID" \
  --name braid-benchmark-pull \
  --format json | jq -r '.metadata.id')

export BENCHMARK_GROUP_ID=$(nebius iam group create \
  --parent-id "$PROJECT_ID" \
  --name braid-benchmark-registry-viewers \
  --format json | jq -r '.metadata.id')

nebius iam group-membership create \
  --parent-id "$BENCHMARK_GROUP_ID" \
  --member-id "$BENCHMARK_SA_ID"

export BENCHMARK_PERMIT_ID=$(nebius iam access-permit create \
  --parent-id "$BENCHMARK_GROUP_ID" \
  --resource-id "$REGISTRY_ID" \
  --role viewer \
  --format json | jq -r '.metadata.id')
```

Build and push from the authenticated developer machine. The credential helper
gets short-lived credentials from the local Nebius profile; no registry token is
written into this repository or passed on the command line:

```bash
nebius registry configure-helper
export IMAGE="$REGISTRY_HOST/${REGISTRY_ID#registry-}/braid-device-gate:$(git -C pgx-mcts-bench rev-parse --short HEAD)"

docker build \
  -f pgx-mcts-bench/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t "$IMAGE" .
docker push "$IMAGE"
```

Create an isolated preemptible L40S VM. A dynamic public IP and managed boot disk
are deleted with the VM. `STOP` preserves the disk if Nebius preempts it, while
`FAIL` prevents an automatic restart from spending beyond the intended gate:

```bash
export BENCHMARK_VM_ID=$(nebius compute instance create \
  --parent-id "$PROJECT_ID" \
  --name braid-gpu-gate \
  --resources-platform gpu-l40s-a \
  --resources-preset 1gpu-8vcpu-32gb \
  --service-account-id "$BENCHMARK_SA_ID" \
  --preemptible-on-preemption stop \
  --recovery-policy fail \
  --boot-disk-managed-disk-name braid-gpu-gate-boot \
  --boot-disk-managed-disk-size-gibibytes 80 \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda12 \
  --network-interfaces "[{\"subnet_id\":\"$SUBNET_ID\",\"ip_address\":{},\"public_ip_address\":{}}]" \
  --cloud-init-user-data "$(cat "$CLOUD_INIT_FILE")" \
  --format json | jq -r '.metadata.id')
```

On the VM, the attached service account obtains a short-lived access token from
instance metadata. This avoids issuing a static key:

```bash
nebius iam get-access-token | \
  sudo docker login "$REGISTRY_HOST" --username iam --password-stdin
sudo docker pull "$IMAGE"
```

## Build

Both repositories must be adjacent in the Docker build context:

```bash
cd /path/containing/rf-knots-and-pgx-mcts-bench
docker build \
  -f pgx-mcts-bench/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t braid-device-gate .
```

Run the container with Nebius's NVIDIA runtime and a durable host directory:

```bash
mkdir -p "$PWD/pgx-mcts-bench/artifacts/nebius-device-gate"
docker run --rm --gpus all \
  -v "$PWD/pgx-mcts-bench/artifacts/nebius-device-gate:/work/pgx-mcts-bench/artifacts/nebius-device-gate" \
  braid-device-gate braid-device-benchmark \
  --devices cpu,cuda \
  --only u1-puct,s-w11-128,s-tape4,s-scan-gru \
  --actor-batches 8,32,64,128 \
  --stage 8 \
  --eval-games 12 \
  --measured-train-steps 96 \
  --torch-threads 1 \
  --cpu-hourly 0.0248 \
  --gpu-hourly 0.7492 \
  --output artifacts/nebius-device-gate
```

`device-benchmark.md` gives phase timings, the 3x engineering gate, and equal-work
cost at the supplied hourly rates. The JSON beside it
contains the exact candidate configuration, parameter count, device, Torch/CUDA
versions, actor batch, and normalized timing.

Before paying for a VM, list the platforms and inspect capacity in the Nebius
console. The relevant IDs are `gpu-l40s-a`, `gpu-l40s-d`, and `cpu-d3`; project
availability and quotas are authoritative.

## Promotion run after the gate

Current-rung checkpoints are atomic and include network, optimizer, replay,
NumPy/Torch/CUDA RNG states, evaluation history, and completed capped rungs.
`--checkpoint-every 1` is the default and is required on preemptible capacity.

If CPU wins, use one worker per candidate/seed and do not let BLAS multiply every
worker into many threads:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pgx-mcts-bench braid-ladder \
  --only <selected-candidates> \
  --workers 14 \
  --device cpu \
  --selfplay-games 8 \
  --checkpoint-every 1 \
  --stop-after 27 \
  --bounds artifacts/bounds.jsonl \
  --output artifacts/rung27
```

If CUDA passes the gate, use the winning actor batch from the report. Run one
candidate process per GPU; separate candidate networks cannot share one inference
batch without an additional inference-server design.

## Retrieve results and tear down

Copy both report files and any resumable checkpoints before deleting the VM:

```bash
rsync -av <ssh-user>@<vm-ip>:/srv/braid/artifacts/nebius-device-gate/ \
  pgx-mcts-bench/artifacts/nebius-device-gate/

nebius compute instance delete --id "$BENCHMARK_VM_ID"
nebius iam access-permit delete --id "$BENCHMARK_PERMIT_ID"

export BENCHMARK_MEMBERSHIP_ID=$(nebius iam group-membership list-members \
  --parent-id "$BENCHMARK_GROUP_ID" --format json | \
  jq -r --arg member "$BENCHMARK_SA_ID" \
    '.memberships[] | select(.spec.member_id == $member) | .metadata.id')
nebius iam group-membership delete --id "$BENCHMARK_MEMBERSHIP_ID"
nebius iam group delete --id "$BENCHMARK_GROUP_ID"
nebius iam service-account delete --id "$BENCHMARK_SA_ID"
```

Do not delete the VM until the copied JSON and Markdown reports open locally.
Never commit a cloud-init file containing an SSH key, Docker configuration,
Nebius CLI profile, access token, static key, or generated artifact directory.

Nebius references: [VM creation](https://docs.nebius.com/compute/virtual-machines/manage),
[preemptible behavior](https://docs.nebius.com/compute/virtual-machines/preemptible),
[registry authentication](https://docs.nebius.com/container-registry/authentication),
and [custom IAM groups](https://docs.nebius.com/iam/authorization/groups/manage).
