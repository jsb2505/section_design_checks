# Reinforced Concrete Test Layout

This folder is grouped by source-domain to keep related tests together:

- `analysis/`: interaction diagrams, viewers, and analysis helpers
- `code_checks/`: EC2 checks and code-check utilities
- `constitutive/`: concrete/steel constitutive model behavior
- `geometry/`: section/rebar geometry and geometry utilities
- `materials/`: concrete, steel, rebar material definitions
- `ndp/`: Nationally Defined Parameter registry and helpers
- `thermal/`: thermal concrete models

If a test file spans multiple domains and one domain is dominant, place it in the dominant domain.
If no dominant domain exists, keep the file at `tests/test_reinforced_concrete/` and document why at the top of the file.
