# scripts/

Repository helper scripts, mainly for packaging and release workflows.

## Layout

- `package/`: Packaging scripts (Python + templates + configuration)
- `gen_pto_isa_capability_manifest.py`: Export a JSON capability manifest for PTO ISA instructions, public intrinsics, and front-end dtype exposure

## Entry Point

- `build.sh --pkg` triggers the packaging flow implemented under `scripts/package/`
- `python3 scripts/gen_pto_isa_capability_manifest.py --output /tmp/pto-isa-capability.json` writes the capability manifest for downstream tooling gates
