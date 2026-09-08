"""Flow Matching robot policy, pluggable into lerobot's lerobot-train/lerobot-eval CLIs.

Importing this package registers "flow_matching" as a lerobot policy type (via
`@PreTrainedConfig.register_subclass` in `configuration_flow_matching`), so lerobot's
`register_third_party_plugins()` -- called automatically by `lerobot-train`/`lerobot-eval` for any
installed distribution named `lerobot_policy_*` -- picks it up with no further setup required.
"""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from e

from .configuration_flow_matching import FlowMatchingConfig
from .modeling_flow_matching import FlowMatchingPolicy
from .processor_flow_matching import make_flow_matching_pre_post_processors

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingPolicy",
    "make_flow_matching_pre_post_processors",
]
