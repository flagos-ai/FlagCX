# vLLM observability examples
ref: vllm/examples/observability

```txt
observability/
├── dashboards/                          # static dashboard definitions (no deploy scripts)
│   ├── README.md
│   ├── grafana/
│   │   ├── README.md
│   │   ├── performance_statistics.json  # Grafana dashboard: performance panels
│   │   └── query_statistics.json        # Grafana dashboard: request/query panels
│   └── perses/
│       ├── README.md
│       ├── performance_statistics.yaml  # Perses equivalent of the same dashboard
│       └── query_statistics.yaml
├── metrics/
│   └── offline.py                       # how to read vLLM metrics in offline (non-server) mode
├── opentelemetry/
│   ├── README.md
│   └── dummy_client.py                  # example client wired up to OTel tracing
└── prometheus_grafana/
    ├── README.md
    ├── docker-compose.yaml              # one-shot Prometheus + Grafana stack
    ├── prometheus.yaml                  # Prometheus scrape config against vLLM /metrics
    └── grafana.json                     # Grafana dashboard for the compose stack
```

# SGLang observability examples
ref: sglang/examples/monitoring

```txt
monitoring/
├── README.md
├── docker-compose.yaml          # one-shot Prometheus + Grafana against a live sglang server
├── tracing_compose.yaml         # separate tracing stack (pairs with opentelemetry.yaml)
├── prometheus.yaml              # Prometheus scrape config for the sglang server /metrics
├── opentelemetry.yaml           # OTel collector config (tracing)
└── grafana/
    ├── datasources/
    │   └── datasource.yaml      # auto-provisions the Prometheus datasource
    └── dashboards/
        ├── config/
        │   └── dashboard.yaml   # dashboard provisioning, points at json/
        └── json/
            └── sglang-dashboard.json   # the only official sglang dashboard
```

## What PTD takes from these

Both upstream setups assume a **live** server being scraped. PTD replays captured
snapshots instead, so only the dashboard JSON is reused:

- `dashboard/vllm/performance_statistics.json`, `query_statistics.json`
- `dashboard/sglang/sglang-dashboard.json`

The compose files, Perses variants, OTel collectors, and scrape configs are not used —
see the repository README for how the offline path is wired instead.
