# G1 pi0.5 Pick Pipeline

These entries capture the corrected G1 action interface used for the fixed-position blue-object pick task.

The model emits nine physical action values:

- seven arm joints use one delta-to-absolute conversion;
- two gripper master joints remain absolute commands.

The training entry validates the fixed-v3 dataset before starting GPU training. The evaluation service loads the resulting checkpoint and exposes nine-dimensional actions to the websocket evaluator.

The scripts depend on the OpenPI training package and the local fixed-v3 dataset. Update the paths in the configuration before running them outside the original server workspace.
