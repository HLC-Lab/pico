# PICO Release Dataset

The benchmark and tracing data used for the PICO evaluation is archived on
Zenodo instead of being stored in the Git repository.

- DOI: 10.5281/zenodo.20796082
- Record: https://zenodo.org/records/20796082
- Archive: pico-release-data-2026-06-22.tar.gz
- Exported from commit: 6afa4115deb77940f7cbbdded02da6fb9531c40c

The archive preserves the repository-relative layout expected by the plotting
and analysis tools:

- results/
- tracer/sinfo/
- tracer/alloc_example/leonardo/

To restore the data in a PICO checkout, download the archive from Zenodo and
extract it at the repository root:

```bash
tar -xzf pico-release-data-2026-06-22.tar.gz -C /path/to/pico --strip-components=1
```

The archive includes `SHA256SUMS`. After extraction, verify file integrity from
the repository root:

```bash
sha256sum -c SHA256SUMS
```

Generated benchmark outputs, metadata CSV files, and tracer data are ignored by
Git. Keep source files such as `results/generate_metadata.py`, result helper
scripts, tracer code, and topology maps in the repository.
