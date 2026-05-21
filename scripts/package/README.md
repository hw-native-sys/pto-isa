# scripts/package/

Packaging and release scripts/templates for PTO Tile Lib. This directory is used to generate distributable artifacts.

## Layout (High Level)

- `pto_isa/`: Package templates and helper scripts specific to PTO Tile Lib
- `module/`: Module-level packaging descriptors
- `common/`: Shared helpers and configuration fragments

## Build and Install

### Packaging

```bash
cd ${git_clone_path}
./build.sh --pkg
```

After the build completes, the `.run` installer package is generated under `scripts/package/output/`.

### Installation

Install via the `.run` self-extracting package. An install type is required (choose one):

| Option | Description |
|--------|-------------|
| `--full` | Full installation: headers, libraries, test resources, and all content |
| `--run` | Runtime installation: only libraries and dependencies needed at runtime |
| `--devel` | Development installation: headers and libraries |

```bash
# Full installation to a custom path
./scripts/package/output/pto_isa_*.run --full --install-path=/your/install/path

# Quiet install, skip interactive prompts (for CI/CD and non-interactive environments)
./scripts/package/output/pto_isa_*.run --full --quiet --install-path=/your/install/path

# Runtime-only installation
./scripts/package/output/pto_isa_*.run --run --install-path=/your/install/path
```

Common install options:

| Option | Description |
|--------|-------------|
| `--install-path=<path>` | Specify installation target directory |
| `--quiet` | Quiet mode, skip interactive confirmation (for non-interactive environments) |
| `--install-for-all` | Install for all users |
| `--uninstall` | Uninstall the installed product |
| `--upgrade` | Upgrade an existing installation |
| `--version` | Query package version information |
| `--pre-check` | Pre-install dependency check |

For a full list of options:

```bash
./scripts/package/output/pto_isa_*.run --help
```
