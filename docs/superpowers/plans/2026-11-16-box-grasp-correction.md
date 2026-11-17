# Box grasp correction implementation plan

> Execute inline with superpowers:executing-plans; preserve existing user work.

**Goal:** Correct the three failures reproduced in the live single-box trials: wrong identity, premature closure, ungrounded lateral adjustment.

**Architecture:** Keep vision for target XY and semantic judgment. For upright right-arm boxes only, use the simulator's existing collision geometry to check identity and compute the open fingertip midpoint. Move in bounded 1 cm increments toward a locked visual XY and geometry-derived grasp height. Never interpret image-relative left/right as world axes for these boxes. Require both geometric readiness and visual grasp authorization to close. Unknown perception still stops. This is a simulation-assisted execution correction, not a real-robot perception implementation.

**Tech stack:** Python in orcalab_9_1, NumPy, existing MuJoCo scene, Euler tools.

**Spec / authorization:** User requested correction after the three live failures documented in docs/agentic_execution_supervision.md. Implementation choices are ours; no new training, dependencies, gain/friction changes or broad collision exemptions. No shipping requested.

Approaches: prompt-only correction improves identity but reproduced empty closure; lowering by a fixed offset does not handle current pose; choose measured fingertip alignment with bounded movement. Keep left bottle behavior unchanged in this bounded repair.

## Tasks

- [x] Add failing geometry tests for too-high fingers, aligned fingers, wrong identity, lateral error and malformed geometry; add supervisor tests proving a premature grasp cannot close and image-relative directions cannot move an aligned box sideways.
- [x] Implement pure box alignment in grasp_alignment.py; expose it through ArmTools. Keep finite checks, 1 cm step, 8 mm XY / 6 mm Z tolerance, at least 20 mm vertical overlap and object fitting between open fingers. Reject identity XY mismatch above 25 mm; preserve IK/collision checks on all actual moves.
- [x] Update actual box descriptions, gate right-box grasp alignment, bound adjustment budget to 10 observations / at most 9 cm motion. Add --objects for isolated trials; retain default full list.
- [x] Run targeted tests / lint, then one live pharyngitis trial with saved observations. Diagnose any new blocker from its exact query without claiming success from unit tests.
- [x] Review changes, update evidence and report the observed outcome and remaining limits.

Review focus: unknown/no image cannot authorize closure; geometric references must use the same identified object; a blocked movement cannot be skipped; both finger pads must overlap the object and clear its width; geometry-assisted behavior must not be reported as model learning.


## Evidence-driven addition

The first corrected trial reached finger contact but could not plan the test lift.
The exact saved query showed two phase-transition defects, now corrected:

- Carried objects retain the existing grasp finger-chain contact permission; table, other objects and non-finger links remain checked.
- A planning-only attachment is expressed at the commanded path start, preserving the observed object world pose. Measured attachment and command continuity stay unchanged. Reject wrist mismatch above 5 mm / 0.035 rad or attachment slip above 10 mm / 0.10 rad.

Regression verification: 78 tests and 33 subtests. The saved failing query now yields a full two-waypoint test-lift path with the existing 1 mm penetration threshold. This replay reconstructs the measured wrist from the saved attachment; it is not a physical lift.

The latest fresh online trial was blocked before robot action by target identity mismatch on a nearly black head-camera image. Full physical lift/placement and the two-bottle sequence remain unverified. Do not mark the four-object task solved.
