# Contributing to FlagCX

Thanks for your interest in contributing to FlagCX! Here's how to get involved.

## Reporting Issues

Open an issue on [GitHub Issues](https://github.com/flagos-ai/FlagCX/issues) with:

- A clear title and description
- Steps to reproduce the problem
- Expected vs. actual behavior
- Environment details (OS, hardware, driver versions)

## Contributing Code

1. Fork the repo and create a branch from `main`.
2. Make your changes — keep commits focused and well-described.
3. Add or update tests in the `test/` directory if applicable.
4. Make sure the project builds cleanly (`make` at the repo root).
5. Open a pull request against `main` with a summary of what changed and why.

Please follow the [Code of Conduct](./CODE_OF_CONDUCT.md) in all interactions.

## Documentation

- [Getting Started](./docs/getting_started.md) — build and setup instructions
- [User Guide](./docs/user_guide.md) — usage and API overview
- [Environment Variables](./docs/environment_variables.md) — runtime configuration
- [Changelog](./docs/CHANGELOG.md) — release history

## Project Structure

```
flagcx/          — core library (adaptor, core, kernels, runner, service)
adaptor_plugin/  — example adaptor plugin implementation and usage
plugin/          — plugins for integrating with upper-level frameworks or applications
test/            — test suites
docs/            — documentation
makefiles/       — build system helpers
packaging/       — packaging scripts
third-party/     — third-party dependencies
```

## License

FlagCX is licensed under the [Apache License 2.0](./LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.
