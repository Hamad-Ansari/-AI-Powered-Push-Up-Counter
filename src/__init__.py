"""AI-Powered Push-Up Counter - application package.

Layers:

* :mod:`src.pose`         - pose backends and the standardized keypoint contract
* :mod:`src.analysis`     - angles, smoothing, movement, posture, form scoring
* :mod:`src.counter`      - push-up state machine and rep validation
* :mod:`src.visualization`- skeleton rendering and the AI fitness HUD
* :mod:`src.video`        - frame sources, the pipeline and video export
* :mod:`src.analytics`    - session tracking, charts and reports
* :mod:`config`           - typed configuration (YAML + environment overrides)
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
