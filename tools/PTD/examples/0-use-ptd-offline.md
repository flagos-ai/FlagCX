# Offline replay

step1. Launch your vllm's/sglang's Prefill/Decode serve, and ensure that you test with vllm bench or real traffic beforehand.

step2. Capture `/metrics` separately for each instance, such as 1p2d shell as below:
> Note: Ensure this step is completed before testing vLLM benchmarks or real traffic.

```bash
while true; do
  echo "# ==== $(date +%s) $(date -Is) ===="
  curl -s http://127.0.0.1:8000/metrics
  echo
  sleep 5
done >> prefill.prom.log 2>&1 &
echo $! > prefill.pid

while true; do
  echo "# ==== $(date +%s) $(date -Is) ===="
  curl -s http://127.0.0.1:9000/metrics
  echo
  sleep 5
done >> decode-1.prom.log 2>&1 &
echo $! > decode-1.pid

while true; do
  echo "# ==== $(date +%s) $(date -Is) ===="
  curl -s http://127.0.0.1:10000/metrics
  echo
  sleep 5
done >> decode-2.prom.log 2>&1 &
echo $! > decode-2.pid
```

step3. Once you have set up the vLLM or sglang PD service, execute the following script.

```bash
kill $(cat prefill.pid) $(cat decode-1.pid) $(cat decode-2.pid)
rm -f prefill.pid decode-1.pid decode-2.pid
```

step3. Put the logs under PTD's `log/` directory and run from the repository root:

```bash
cd /path/to/PTD
PREFILL='log/run/prefill.prom.log' \
DECODE='log/run/decode-1.prom.log,log/run/decode-2.prom.log' \
DASH=ptd-sglang \
docker compose up -d
```

step4. click Grafana on: <http://localhost:3000>. Prometheus on: <http://localhost:9090>.

step5.
Check the conversion result:

```bash
docker compose logs prepare
```
