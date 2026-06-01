#!/bin/bash
set -e

BS=$1
TRITON_HOST=$2
MODEL="higgsInteractionNet"
GRPC_URL="${TRITON_HOST}:8001"
HTTP_URL="${TRITON_HOST}:8000"

SDK_SIF="/home/tardis/SuperSONIC/benchmarks/triton-sdk.sif"
OUTDIR="/home/tardis/SuperSONIC/benchmarks/results/benchmark_htc"
mkdir -p ${OUTDIR}

echo "=== BareHTCondor perf_analyzer ==="
echo "  Batch size:  ${BS}"
echo "  Triton gRPC: ${GRPC_URL}"
echo "  Model:       ${MODEL}"
echo ""

echo "=== Checking Triton health ==="
curl -sf http://${HTTP_URL}/v2/health/ready \
  && echo "Triton is READY" \
  || { echo "ERROR: Triton not ready"; exit 1; }

echo "=== Running perf_analyzer BS=${BS} ==="
singularity exec \
  ${SDK_SIF} \
  /workspace/install/bin/perf_analyzer \
  -i grpc \
  -m ${MODEL} \
  -u ${GRPC_URL} \
  -b ${BS} \
  --concurrency-range 1:1 \
  --measurement-interval 10000 \
  --async \
  --input-data random \
  --verbose-csv \
  -f /tmp/perf_bs${BS}.csv \
  2>&1

# Copy from /tmp to the actual output directory
cp /tmp/perf_bs${BS}.csv ${OUTDIR}/perf_bs${BS}.csv

echo "=== CSV content ==="
cat ${OUTDIR}/perf_bs${BS}.csv