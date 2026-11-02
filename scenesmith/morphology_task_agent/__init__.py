"""Morphology Task Agent: selects the best RL training task from a TaskSmith scene.

Single-shot LLM call that reads rl_artifacts.json, reward_spec.yaml, and
curriculum_spec.yaml to determine which object and task phases are best suited
for morphological hand training (approach → grasp → lift).

Output: morphology_task_config.json in the scene directory, consumed by
SceneSmithHandEnv in the optimal_morphology_rl training pipeline.
"""
