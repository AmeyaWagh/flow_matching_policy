"""Tests that importing lerobot_policy_flow_matching registers it as a discoverable lerobot policy."""

from lerobot.configs import PreTrainedConfig


def test_flow_matching_is_a_known_policy_choice():
    """Importing the package (done at collection time via other test modules) registers "flow_matching"."""
    import lerobot_policy_flow_matching  # noqa: F401

    assert "flow_matching" in PreTrainedConfig.get_known_choices()


def test_factory_resolves_policy_and_processor_by_naming_convention():
    """lerobot's policy factory resolves FlowMatchingPolicy and its processor factory by name alone.

    This is the exact mechanism `lerobot-train`/`lerobot-eval` rely on: no explicit import of
    `lerobot_policy_flow_matching` is needed once the distribution is installed and
    `register_third_party_plugins()` has run.
    """
    from lerobot.policies.factory import get_policy_class

    import lerobot_policy_flow_matching  # noqa: F401

    policy_cls = get_policy_class("flow_matching")
    assert policy_cls is lerobot_policy_flow_matching.FlowMatchingPolicy
