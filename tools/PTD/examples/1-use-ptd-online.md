# Online monitoring

Linux host networking only. Configure one address for each P/D instance:

```bash
DASH=ptd-sglang \
TARGETS=127.0.0.1:8000,127.0.0.1:8001 \
docker compose --profile online up -d
```

The servers must expose `/metrics`. Check an endpoint first:

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

Prometheus target status: <http://localhost:9090/targets>.
